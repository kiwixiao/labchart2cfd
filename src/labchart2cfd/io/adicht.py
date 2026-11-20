#!/usr/bin/env python
"""Pure Python reader for native LabChart recordings (.adicht).

No public description of this format exists, so the layout was decoded from
the bytes of real recordings and then checked sample for sample against
LabChart's own MATLAB exports of the same recordings. FORMAT NOTES at the
bottom of this file describe the layout and say what was verified.

The reader checks every structural constant it relies on. It raises
AdichtFormatError on anything it has not been verified on, so it does not
return numbers from a layout it does not understand.

LabChart does not store computed channels (channel calculations) in the
file. It recomputes them on display and on export. This reader finds the
channels that carry a calculation in the file's settings. It reproduces one
kind, the Integral calculation, with arithmetic that matched LabChart's
export bit for bit. A channel with any other calculation is left empty.
See FORMAT NOTES, sections 5 to 7.

Reading needs numpy only. Writing the .mat needs scipy.

Command line:
    python adicht_reader.py <file.adicht> --info
        prints records, channels and comments and decodes nothing
    python adicht_reader.py <file.adicht> --out <dir>
        writes <stem>.mat in LabChart's "Export MATLAB" layout (MAT version 4,
        single precision) and <stem>.json with the metadata
    options for a partial export, as in LabChart's export dialog:
        --channels 1,3,5              channel numbers, 1 based
        --record N --start S --end E  one record, S to E seconds from its start
"""

import argparse
import datetime
import json
import mmap
import struct
import sys
from pathlib import Path

import numpy as np

__all__ = ["AdichtFormatError", "AdichtFile", "plan_export", "build_export", "write_outputs"]


class AdichtFormatError(ValueError):
    """The file does not match a layout this reader was verified on."""


# --------------------------------------------------------------------------
# Container layer: "ADInstruments Compound File"
# --------------------------------------------------------------------------

_SIGNATURE = "ADInstruments Compound File"
_HEADER_SIZE = 0x200
_DIR_ENTRY_SIZE = 160
_DIR_NAME_BYTES = 72
_KIND_STORAGE = 1
_KIND_STREAM = 2
_STREAM_MAGICS = (bytes.fromhex("aaaa1355"), bytes.fromhex("aaaa1555"))
_BLOCKS_MAGIC = bytes.fromhex("555503aa")


class _DirEntry:
    __slots__ = ("name", "first", "size", "last", "kind")

    def __init__(self, name, first, size, last, kind):
        self.name = name
        self.first = first
        self.size = size
        self.last = last
        self.kind = kind


class _CompoundFile:
    """Directory tree and block chains of the container (see FORMAT NOTES 1)."""

    def __init__(self, mm):
        self.mm = mm
        if len(mm) < _HEADER_SIZE:
            raise AdichtFormatError("File is shorter than the 512 byte header.")
        sig = bytes(mm[: 2 * len(_SIGNATURE)]).decode("utf-16-le", errors="replace")
        if sig != _SIGNATURE:
            raise AdichtFormatError("Not an ADInstruments Compound File (signature %r)." % sig)
        words = struct.unpack_from("<14I", mm, 0x80)
        self.header_words = words
        if words[1] != _HEADER_SIZE:
            raise AdichtFormatError("Unexpected header size word %#x." % words[1])
        shift = words[6]
        if not 9 <= shift <= 20:
            raise AdichtFormatError("Unexpected block size shift %d." % shift)
        self.block_size = 1 << shift
        body = len(mm) - _HEADER_SIZE
        if body % self.block_size:
            raise AdichtFormatError(
                "File size is not the header plus a whole number of blocks. "
                "The file is truncated or not fully downloaded."
            )
        self.n_blocks = body // self.block_size + 2
        self.table_first = words[11]
        if not 2 <= self.table_first < self.n_blocks:
            raise AdichtFormatError("Block table start %d is out of range." % self.table_first)
        self.table_offset = self._offset(self.table_first)
        per_block = self.block_size // 8
        capacity = words[7]
        if capacity == 0 or capacity % per_block or words[9] != (-capacity) & 0xFFFFFFFF:
            raise AdichtFormatError(
                "Unexpected block table capacity words %#x, %#x." % (words[7], words[9])
            )
        self.table_capacity = capacity
        table_blocks = capacity // per_block
        if self.table_first + table_blocks > self.n_blocks:
            raise AdichtFormatError("Block table runs past the end of the file.")
        if struct.unpack_from("<I", mm, self.table_offset + 4)[0] != 0x80000000:
            raise AdichtFormatError("Block table does not start with its marker entry.")
        # The table is addressed as one contiguous array, so its own blocks
        # must be consecutive. Its chain is stored in the table like any other.
        for k in range(table_blocks):
            b = self.table_first + k
            nxt, prev = self._next(b)
            want_next = b + 1 if k + 1 < table_blocks else 0
            want_prev = b - 1 if k else 0
            if nxt != want_next or prev != want_prev:
                raise AdichtFormatError(
                    "Block table is not stored in consecutive blocks (block %d links %d, %d)."
                    % (b, nxt, prev)
                )
        self.root = self._entry_at(mm, 0xB8)
        if self.root.name != "ADIRootStorage" or self.root.kind != _KIND_STORAGE:
            raise AdichtFormatError("Root directory entry not found at 0xB8.")

    def _offset(self, block):
        if not 2 <= block < self.n_blocks:
            raise AdichtFormatError("Block number %d is out of range." % block)
        return _HEADER_SIZE + (block - 2) * self.block_size

    def _next(self, block):
        if not 0 <= block < self.table_capacity:
            raise AdichtFormatError("Block %d is outside the block table." % block)
        return struct.unpack_from("<II", self.mm, self.table_offset + 8 * block)

    @staticmethod
    def _entry_at(buf, o):
        name = bytes(buf[o : o + _DIR_NAME_BYTES]).decode("utf-16-le", errors="replace")
        name = name.split("\0", 1)[0]
        first, size, last = struct.unpack_from("<3Q", buf, o + _DIR_NAME_BYTES)
        kind = struct.unpack_from("<I", buf, o + _DIR_NAME_BYTES + 0x44)[0]
        if kind not in (_KIND_STORAGE, _KIND_STREAM):
            raise AdichtFormatError("Directory entry %r has unknown kind %d." % (name, kind))
        return _DirEntry(name, first, size, last, kind)

    def chain(self, entry):
        """Blocks of a stream in order, following and checking the table."""
        if entry.size == 0:
            return []
        need = -(-entry.size // self.block_size)
        blocks = [entry.first]
        b = entry.first
        self._offset(b)
        while b != entry.last:
            if len(blocks) >= need:
                raise AdichtFormatError(
                    "Block chain of %r is longer than its size needs." % entry.name
                )
            nxt, _prev = self._next(b)
            self._offset(nxt)
            if self._next(nxt)[1] != b:
                raise AdichtFormatError(
                    "Block chain of %r is inconsistent at block %d." % (entry.name, b)
                )
            blocks.append(nxt)
            b = nxt
        if len(blocks) != need:
            raise AdichtFormatError(
                "Stream %r needs %d blocks but its chain has %d." % (entry.name, need, len(blocks))
            )
        return blocks

    def read(self, entry):
        """Whole stream as bytes. Use only for small streams."""
        parts = [self.mm[self._offset(b) : self._offset(b) + self.block_size]
                 for b in self.chain(entry)]
        return b"".join(parts)[: entry.size]

    def listdir(self, entry):
        if entry.kind != _KIND_STORAGE:
            raise AdichtFormatError("%r is not a storage." % entry.name)
        if entry.size == 0:
            return []
        d = self.read(entry)
        n, neg = struct.unpack_from("<ii", d, 0)
        if n < 0 or neg != -n or len(d) != 8 + _DIR_ENTRY_SIZE * n:
            raise AdichtFormatError("Directory %r is malformed." % entry.name)
        return [self._entry_at(d, 8 + _DIR_ENTRY_SIZE * i) for i in range(n)]

    def child(self, entry, name):
        found = [e for e in self.listdir(entry) if e.name == name]
        if len(found) != 1:
            raise AdichtFormatError(
                "Expected exactly one %r in %r, found %d." % (name, entry.name, len(found))
            )
        return found[0]


class _StreamView:
    """Random access to a large stream without copying it whole."""

    def __init__(self, cf, entry):
        self.cf = cf
        self.size = entry.size
        self.blocks = cf.chain(entry)
        self.contiguous = self.blocks == list(range(entry.first, entry.first + len(self.blocks)))
        self.base = cf._offset(entry.first) if self.blocks else 0

    def read(self, pos, n):
        if pos < 0 or n < 0 or pos + n > self.size:
            raise AdichtFormatError(
                "Read of %d bytes at %d is outside a stream of %d bytes." % (n, pos, self.size)
            )
        if self.contiguous:
            return self.cf.mm[self.base + pos : self.base + pos + n]
        bs = self.cf.block_size
        out = bytearray()
        while n:
            bi, bo = divmod(pos, bs)
            take = min(n, bs - bo)
            o = self.cf._offset(self.blocks[bi]) + bo
            out += self.cf.mm[o : o + take]
            pos += take
            n -= take
        return bytes(out)


# --------------------------------------------------------------------------
# Serialised metadata: the "Map" stream (see FORMAT NOTES 2)
# --------------------------------------------------------------------------

_DEF_FLAG = 0x80000000
_STRING_TAG = 0x80000001
_CHUNK_TAG = 0x80020001
_CHANNEL_SIGNATURE = bytes.fromhex("0100001b80")
_CHUNK_INDEX_DTYPE = np.dtype([
    ("tag", "<u4"), ("a", "<u8"), ("n", "<u4"), ("b", "<u4"),
    ("min", "<f4"), ("max", "<f4"), ("offset", "<u8")])

# Field values that occurred in recordings whose decoded samples matched a
# LabChart export bit for bit. Anything else is refused. The meaning of the
# flag words is not known. They did not change the numbers in any export.
_VERIFIED_MAP_VERSIONS = ((3, 5), (3, 7))
_VERIFIED_RECORD_FLAGS = (0, 0x30000000)
_VERIFIED_INDEX_FLAGS = (0, 0x80000000)
_VERIFIED_UNIT_BITS = ((0, 0x0000), (1, 0x4000), (1, 0x8000))   # (conversion on, flag bits)
_VERIFIED_DISPLAY = ((3, 1), (1, 7), (1, 1))
_VERIFIED_DISPLAY2 = ((2, 1), (2, 8), (3, 8))


class _Cursor:
    def __init__(self, buf, name):
        self.buf = buf
        self.pos = 0
        self.name = name

    def fail(self, msg):
        lo = max(0, self.pos - 16)
        ctx = bytes(self.buf[lo : self.pos + 32]).hex(" ")
        raise AdichtFormatError(
            "%s stream, offset %#x: %s (bytes from %#x: %s)" % (self.name, self.pos, msg, lo, ctx)
        )

    def take(self, fmt):
        s = struct.Struct("<" + fmt)
        if self.pos + s.size > len(self.buf):
            self.fail("unexpected end of stream")
        v = s.unpack_from(self.buf, self.pos)
        self.pos += s.size
        return v if len(v) > 1 else v[0]

    def raw(self, n):
        if self.pos + n > len(self.buf):
            self.fail("unexpected end of stream")
        v = bytes(self.buf[self.pos : self.pos + n])
        self.pos += n
        return v

    def expect(self, fmt, *want):
        at = self.pos
        got = self.take(fmt)
        got_t = got if isinstance(got, tuple) else (got,)
        if got_t != want:
            self.pos = at
            self.fail("expected %r, found %r" % (want, got_t))

    def one_of(self, fmt, allowed, what):
        at = self.pos
        got = self.take(fmt)
        if got not in allowed:
            self.pos = at
            self.fail("%s %r was not verified against a LabChart export" % (what, got))
        return got

    def stream_magic(self):
        at = self.pos
        got = self.raw(4)
        if got not in _STREAM_MAGICS:
            self.pos = at
            self.fail("unknown stream magic %s" % got.hex(" "))
        return got

    def count(self):
        """Array header: i32 count followed by i32 -count."""
        at = self.pos
        n, neg = self.take("ii")
        if n < 0 or neg != -n:
            self.pos = at
            self.fail("malformed array header (%d, %d)" % (n, neg))
        return n

    def string(self):
        """u32 0x80000001, array header (length in UTF-16 units with NUL), text."""
        at = self.pos
        tag = self.take("I")
        if tag != _STRING_TAG:
            self.pos = at
            self.fail("expected a string tag, found %#x" % tag)
        n = self.count()
        if n < 1:
            self.pos = at
            self.fail("string without terminator")
        raw = self.raw(2 * n)
        if raw[-2:] != b"\0\0":
            self.pos = at
            self.fail("string is not NUL terminated")
        return raw[:-2].decode("utf-16-le")


class _Scaling:
    """ADC counts to physical units for one channel in one record."""

    __slots__ = ("adc_scale", "adc_offset", "adc_max", "adc_min", "base_unit",
                 "unit_scale", "unit_offset", "units_conversion", "unit_bits",
                 "unit_name", "calibration", "display", "display2")

    @property
    def unit(self):
        return self.unit_name if self.units_conversion else self.base_unit

    @property
    def scale_units(self):
        """Physical units per ADC count (LabChart's "scaleunits")."""
        return self.adc_scale * (self.unit_scale if self.units_conversion else 1.0)

    @property
    def scale_offset(self):
        """Offset in ADC counts, added before scaling (LabChart's "scaleoffset")."""
        if self.units_conversion:
            return self.unit_offset / self.scale_units
        return 0.0


def _parse_scaling(c):
    s = _Scaling()
    c.expect("II", 5, 2)
    s.adc_scale, s.adc_offset = c.take("dd")
    s.adc_max, s.adc_min = c.take("ff")
    if s.adc_offset != 0.0 or (s.adc_min, s.adc_max) != (-32767.0, 32767.0):
        c.fail("ADC offset %r or range (%r, %r) was not verified against a LabChart export"
               % (s.adc_offset, s.adc_min, s.adc_max))
    if not (np.isfinite(s.adc_scale) and s.adc_scale > 0):
        c.fail("invalid ADC scale %r" % s.adc_scale)
    s.display = c.one_of("II", _VERIFIED_DISPLAY, "display format")
    c.expect("H", 1)
    s.base_unit = c.string()
    s.unit_scale, s.unit_offset = c.take("dd")
    s.display2 = c.one_of("II", _VERIFIED_DISPLAY2, "units display format")
    c.expect("H", 0)
    at = c.pos
    conv, bits = c.take("IH")
    if (conv, bits) not in _VERIFIED_UNIT_BITS:
        c.pos = at
        c.fail("units conversion flags (%d, %#x) were not verified against a LabChart export"
               % (conv, bits))
    s.units_conversion = bool(conv)
    s.unit_bits = bits
    s.unit_name = c.string()
    s.calibration = [c.string() for _ in range(4)]
    c.expect("B", 0)
    if s.units_conversion and not (np.isfinite(s.unit_scale) and s.unit_scale != 0
                                   and np.isfinite(s.unit_offset)):
        c.fail("invalid units conversion %r, %r" % (s.unit_scale, s.unit_offset))
    return s


def _parse_source(c):
    version = c.take("I")
    if version == 2:
        cls = c.string()
        if cls not in ("ChannelDataSource", ""):
            c.fail("unknown data source class %r" % cls)
    elif version != 1:
        c.pos -= 4
        c.fail("unknown data source version %d" % version)
    return c.string()   # the channel's hardware setup as XML text


def _shared_object(c, objects, kind, parse):
    """An object defined in place (id | 0x80000000, body) or referenced by id."""
    at = c.pos
    ref = c.take("I")
    oid = ref & ~_DEF_FLAG
    if oid == 0:
        c.pos = at
        c.fail("null %s object" % kind)
    if ref & _DEF_FLAG:
        if oid in objects:
            c.pos = at
            c.fail("object %d defined twice" % oid)
        objects[oid] = (kind, parse(c))
    if oid not in objects:
        c.pos = at
        c.fail("reference to undefined object %d" % oid)
    if objects[oid][0] != kind:
        c.pos = at
        c.fail("object %d is a %s, expected a %s" % (oid, objects[oid][0], kind))
    return objects[oid][1]


def _parse_record(c, idx, objects, chunk_base_min):
    rec = {"index": idx}
    rec_version = c.take("I")
    if rec_version not in (1, 3):
        c.pos -= 4
        c.fail("unknown record version %d" % rec_version)
    tick_num, tick_den = c.take("QQ")
    if tick_num != 1 or tick_den == 0:
        c.fail("unexpected tick period %d / %d s" % (tick_num, tick_den))
    rec["tick_rate"] = float(tick_den)
    rec["start_local_epoch"] = c.take("d")
    rec["utc_offset_s"] = c.take("d")
    at = c.pos
    crop = c.take("ii")
    if crop != (0, 0):
        # Seen as (-X, +X) in files saved from a selection of a longer
        # recording. No LabChart export of such a file was available to check
        # how X enters the block start time, so these files are refused.
        c.pos = at
        c.fail("record %d carries a start offset (%d, %d ticks). Files saved from a "
               "selection were not verified against a LabChart export" % (idx + 1, crop[0], crop[1]))
    c.expect("I", 0)
    chunk_base = c.take("I")
    if chunk_base < chunk_base_min:
        c.fail("chunk base %d is below the %d chunks already indexed" % (chunk_base, chunk_base_min))
    rec["n_ticks"] = c.take("Q")
    if rec_version >= 3:
        rec["name"] = c.string()
        c.expect("I", 0)
        rec["flags"] = c.one_of("I", _VERIFIED_RECORD_FLAGS, "record flags")
    else:
        rec["name"] = ""
        rec["flags"] = 0

    # Tick chunks, with the comments that fall inside each chunk.
    n_tick_chunks = c.count()
    comments = []
    tick_total = 0
    n_data_chunks = 0
    for _ in range(n_tick_chunks):
        c.expect("I", 3)
        n_in_chunk = c.take("I")
        c.expect("H", 0)
        flags = c.take("H")
        if flags & 0x3FFF:
            c.fail("unknown tick chunk flags %#x" % flags)
        c.expect("H", 0)
        n_com = c.take("H")
        for _ in range(n_com):
            c.expect("II", 2, 0)
            number = c.take("I")
            channel, tick = c.take("hH")
            text = c.string()
            if tick >= max(n_in_chunk, 1) or channel < -1:
                c.fail("comment at tick %d, channel %d does not fit its chunk of %d ticks"
                       % (tick, channel, n_in_chunk))
            comments.append({"number": number, "channel": channel,
                             "tick": tick_total + tick, "text": text})
        tick_total += n_in_chunk
        n_data_chunks += 1 if n_in_chunk else 0
    if tick_total != rec["n_ticks"]:
        c.fail("tick chunks sum to %d, record says %d" % (tick_total, rec["n_ticks"]))
    rec["comments"] = comments

    channels = []
    for ch in range(c.count()):
        channels.append(_parse_channel(c, ch, idx, objects, tick_den, rec["n_ticks"]))
    for com in comments:
        if com["channel"] >= len(channels):
            c.fail("comment refers to channel %d of %d" % (com["channel"] + 1, len(channels)))
    rec["channels"] = channels
    return rec, chunk_base + n_data_chunks


def _parse_channel(c, ch, rec_idx, objects, tick_den, n_ticks):
    at = c.pos
    if c.raw(len(_CHANNEL_SIGNATURE)) != _CHANNEL_SIGNATURE:
        c.pos = at
        c.fail("channel record signature not found")
    ch_version = c.take("I")
    if ch_version not in (2, 3):
        c.pos -= 4
        c.fail("unknown channel record version %d" % ch_version)
    c.expect("I", ch)
    at = c.pos
    if c.take("I") != 1:
        c.pos = at
        c.fail("channel %d has a sampling layout that is not decoded "
               "(seen in recordings imported from other systems)" % (ch + 1))
    c.expect("IQIIffQ", _CHUNK_TAG, 1, 1, 0, 0.0, 0.0, 1)
    rate_den = c.take("Q")
    if rate_den != tick_den:
        c.fail("channel %d is sampled at %d Hz but the record ticks at %d Hz. "
               "Channels at a lower rate are not decoded" % (ch + 1, rate_den, tick_den))
    # Two shared objects follow. Ids come from one counter for both kinds.
    scaling = _shared_object(c, objects, "scaling", _parse_scaling)
    if ch_version >= 3:
        c.expect("I", 1)
    setup_xml = _shared_object(c, objects, "source", _parse_source)
    n_samples = c.take("Q")
    # Version 3: a flag word, 0 or 0x80000000. Version 2: the 0 based record number.
    allowed = _VERIFIED_INDEX_FLAGS if ch_version >= 3 else (rec_idx,)
    index_flags = c.one_of("I", allowed, "chunk index flags")
    at = c.pos
    longest, chunk_len = c.take("II")
    if chunk_len == 0 or longest > chunk_len:
        c.pos = at
        c.fail("unexpected chunk lengths %d, %d" % (longest, chunk_len))
    n_chunks = c.count()
    entries = np.frombuffer(c.raw(_CHUNK_INDEX_DTYPE.itemsize * n_chunks), dtype=_CHUNK_INDEX_DTYPE)
    if n_chunks:
        if (np.any(entries["tag"] != _CHUNK_TAG) or np.any(entries["a"] != 0)
                or np.any(entries["b"] != 0)):
            c.fail("chunk index of channel %d holds an unknown entry type" % (ch + 1))
        if int(entries["n"].max()) != longest or int(entries["n"].min()) < 1:
            c.fail("chunk index of channel %d disagrees with its longest chunk %d"
                   % (ch + 1, longest))
    if int(entries["n"].sum(dtype=np.uint64)) != n_samples:
        c.fail("chunk index of channel %d does not sum to its %d samples" % (ch + 1, n_samples))
    if n_samples != n_ticks:
        c.fail("channel %d has %d samples for %d ticks. Channels that do not span the "
               "whole record are not decoded" % (ch + 1, n_samples, n_ticks))
    return {"version": ch_version, "index_flags": index_flags, "scaling": scaling,
            "setup_xml": setup_xml, "n_samples": int(n_samples), "chunk_len": int(chunk_len),
            "chunks": entries}


def _parse_map(buf):
    c = _Cursor(buf, "Map")
    magic = c.stream_magic()
    c.expect("IQ", 2, 0)
    ident = c.raw(c.take("H"))
    if ident != b"ADInstruments Chart Data\0":
        c.fail("unexpected identifier %r" % ident)
    version = c.one_of("HH", _VERIFIED_MAP_VERSIONS, "Map version")
    n_channels = c.take("H")
    c.expect("B", 0)
    objects = {}       # shared object table: id -> (kind, object)
    records = []
    chunk_base_min = 0
    while True:
        at = c.pos
        idx = c.take("i")
        if idx == -1:
            break
        if idx != len(records):
            c.pos = at
            c.fail("record index %d out of sequence (expected %d)" % (idx, len(records)))
        c.expect("I", 0)
        rec, chunk_base_min = _parse_record(c, idx, objects, chunk_base_min)
        if len(rec["channels"]) != n_channels:
            c.fail("record %d holds %d channels, the document has %d. Records that lack "
                   "a channel were not verified against a LabChart export"
                   % (idx + 1, len(rec["channels"]), n_channels))
        records.append(rec)
    titles = [c.string() for _ in range(c.count())]
    total_unpacked, blocks_end = c.take("QQ")
    trailer = c.string() if c.pos < len(buf) else None   # absent in the older layout
    if c.pos != len(buf):
        c.fail("%d unread bytes at the end" % (len(buf) - c.pos))
    return {"version": version, "n_channels": n_channels, "stream_magic": magic.hex(),
            "records": records,
            "titles": titles, "total_unpacked": total_unpacked, "blocks_end": blocks_end,
            "trailer": trailer}


# --------------------------------------------------------------------------
# Channel calculations: Settings/SettingsRegistrar (see FORMAT NOTES 5)
# --------------------------------------------------------------------------

_CALC_TAG = bytes.fromhex("23000c8000000000")
_INTEGRAL_GUID = bytes.fromhex("104b96233290d411811100a0c9e26433")
_INTEGRAL_RECORD = struct.Struct("<IHHIIiddIB")
_INTEGRAL_TEXT = "sta\toff"
_VERIFIED_INTEGRAL_VERSIONS = (6, 7, 8, 9, 10, 11)


def _read_calculations(cf):
    """Channels that carry a LabChart channel calculation.

    Returns {channel index: {"kind": "integral", "source": index} or
    {"kind": "unknown"}}. The stream is scanned for the 8 byte tag that
    precedes every per channel calculation entry. An entry is accepted as an
    Integral only when every field equals a value seen in recordings whose
    Integral channel was reproduced bit for bit. Anything else is "unknown".
    """
    settings = cf.child(cf.root, "Settings")
    d = cf.read(cf.child(settings, "SettingsRegistrar"))
    calcs = {}
    pos = d.find(_CALC_TAG)
    while pos >= 0:
        o = pos + len(_CALC_TAG)
        if o + 8 > len(d):
            raise AdichtFormatError("Truncated channel calculation entry in the settings.")
        channel = struct.unpack_from("<I", d, o)[0]
        entry = {"kind": "unknown"}
        o += 8
        if d[o : o + 16] == _INTEGRAL_GUID and o + 16 + _INTEGRAL_RECORD.size + 12 <= len(d):
            oid, one, version, f1, source, neg, a, b, two, flag = _INTEGRAL_RECORD.unpack_from(d, o + 16)
            c = _Cursor(d, "SettingsRegistrar")
            c.pos = o + 16 + _INTEGRAL_RECORD.size
            try:
                text = c.string()
            except AdichtFormatError:
                text = None
            if (oid & _DEF_FLAG and one == 1 and version in _VERIFIED_INTEGRAL_VERSIONS
                    and f1 - channel in (0, 1) and neg == -1 and a == 1.0 and b == -1.0
                    and two in (2, 3) and flag == 1 and text == _INTEGRAL_TEXT):
                entry = {"kind": "integral", "source": source}
        if channel in calcs:
            entry = {"kind": "unknown"}      # two entries for one channel: not seen, not trusted
        calcs[channel] = entry
        pos = d.find(_CALC_TAG, pos + 1)
    return calcs


# --------------------------------------------------------------------------
# Sample chunks: the "Blocks_0" stream (see FORMAT NOTES 3)
# --------------------------------------------------------------------------

_N_PLANES = 17
_CHUNK_HEADER = struct.Struct("<IIiHHI")  # length, 0, -length, 0x0104, 0, unpacked bytes


def _unpack_rle(payload, expected):
    """Run length decoding (PackBits like, but 0x80 is a run of 129).

    Control byte c < 0x80: copy the next c + 1 bytes (1 to 128).
    Control byte c >= 0x80: repeat the next byte 257 - c times (2 to 129).
    The unpacked size must equal `expected` exactly.
    """
    out = np.empty(expected, dtype=np.uint8)
    src = np.frombuffer(payload, dtype=np.uint8)
    i = 0
    o = 0
    n = len(payload)
    while i < n:
        ctl = payload[i]
        i += 1
        if ctl < 0x80:
            k = ctl + 1
            if i + k > n or o + k > expected:
                raise AdichtFormatError("Run length data overruns its chunk.")
            out[o : o + k] = src[i : i + k]
            i += k
        else:
            k = 257 - ctl
            if i >= n or o + k > expected:
                raise AdichtFormatError("Run length data overruns its chunk.")
            out[o : o + k] = payload[i]
            i += 1
        o += k
    if o != expected:
        raise AdichtFormatError(
            "Run length data unpacked to %d bytes, expected %d." % (o, expected)
        )
    return out


def _decode_chunk(stream, offset, n_samples):
    """Decode one chunk. Returns (ADC counts as int64, bytes used in Blocks_0)."""
    length, zero, neg, kind, zero2, unpacked = _CHUNK_HEADER.unpack(stream.read(offset, 20))
    if zero != 0 or neg != -length or kind != 0x0104 or zero2 != 0 or length < 16:
        raise AdichtFormatError("Unknown chunk header at Blocks_0 offset %d." % offset)
    if n_samples < 1 or unpacked != 2 * n_samples:
        raise AdichtFormatError(
            "Chunk at %d declares %d unpacked bytes for %d samples." % (offset, unpacked, n_samples)
        )
    plane_bytes = 4 * -(-n_samples // 32)
    payload = stream.read(offset + 20, length - 16)
    planes = _unpack_rle(payload, _N_PLANES * plane_bytes).reshape(_N_PLANES, plane_bytes)
    bits = np.unpackbits(planes, axis=1, bitorder="little")
    if bits[:, n_samples:].any():
        raise AdichtFormatError("Chunk at %d has data in its padding bits." % offset)
    bits = bits[:, :n_samples]
    mag = np.zeros(n_samples, dtype=np.int32)
    for p in range(1, _N_PLANES):
        if planes[p].any():
            mag |= bits[p].astype(np.int32) << (p - 1)
    sign = bits[0].astype(bool)
    if np.any(sign & (mag == 0)):
        raise AdichtFormatError("Chunk at %d holds a negative zero difference." % offset)
    return np.cumsum(np.where(sign, -mag, mag), dtype=np.int64), 4 + length


# --------------------------------------------------------------------------
# Public reader
# --------------------------------------------------------------------------

_MATLAB_DATENUM_OF_UNIX_EPOCH = 719529.0


class AdichtFile:
    """A decoded .adicht recording. Channel and record indices are 0 based.

    Attributes:
        software          LabChart version strings stored in the file
        n_channels        recorded channels
        n_records         records (LabChart calls them blocks)
        titles            titles of the recorded channels
        all_titles        every title in the file, unrecorded channels included
        tick_rate[r]      Hz
        sample_rate[c][r] Hz (equal to the tick rate, lower rates are refused)
        n_samples[c][r]
        units[c][r]
        start_local[r]    record start as a naive datetime in the recording
                          computer's local time
        utc_offset_s[r]   seconds to add to local time to get UTC (inferred)
        comments          dicts with record, channel (-1 for all channels),
                          tick, time_s, number, text
    Methods:
        adc(c, r)         raw ADC counts of the recorded input, int16
        input_data(c, r)  the recorded input in physical units, float64
        data(c, r)        the channel as LabChart exports it, float64
        scaling(c, r)     dict with the conversion constants of the input
        channel_kind[c]   "recorded", "integral" (recomputed from
                          integral_source[c]) or "calculation" (left out)
    """

    def __init__(self, path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        try:
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except ValueError:
            self._fh.close()
            raise AdichtFormatError("Empty file.")
        try:
            self._load()
        except Exception:
            self.close()
            raise

    def close(self):
        self._blocks = None
        self._cf = None
        try:
            self._mm.close()
        finally:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _load(self):
        cf = _CompoundFile(self._mm)
        self._cf = cf
        data_streams = [e.name for e in cf.listdir(cf.root) if e.name.startswith("Blocks_")]
        if data_streams != ["Blocks_0"]:
            raise AdichtFormatError(
                "Expected exactly one sample stream named Blocks_0, found %r." % data_streams
            )
        self.software = self._read_version(cf)
        meta = _parse_map(cf.read(cf.child(cf.root, "Map")))
        self._meta = meta
        self._blocks = _StreamView(cf, cf.child(cf.root, "Blocks_0"))
        if self._blocks.size < 4 or bytes(self._blocks.read(0, 4)) != _BLOCKS_MAGIC:
            raise AdichtFormatError("Blocks_0 does not start with its magic bytes.")
        if meta["blocks_end"] + 4 != self._blocks.size:
            raise AdichtFormatError(
                "Map says the sample data ends at %d, Blocks_0 holds %d bytes. Seen in files "
                "from which records were deleted, which were not verified."
                % (meta["blocks_end"] + 4, self._blocks.size)
            )
        records = meta["records"]
        self.n_records = len(records)
        self.n_channels = meta["n_channels"] if records else 0
        self.all_titles = meta["titles"]
        if len(self.all_titles) < self.n_channels:
            raise AdichtFormatError("Fewer titles than recorded channels.")
        self.titles = self.all_titles[: self.n_channels]
        self.tick_rate = [r["tick_rate"] for r in records]
        self.n_ticks = [int(r["n_ticks"]) for r in records]
        self.start_local_epoch = [r["start_local_epoch"] for r in records]
        self.utc_offset_s = [r["utc_offset_s"] for r in records]
        epoch = datetime.datetime(1970, 1, 1)
        self.start_local = [epoch + datetime.timedelta(seconds=t) for t in self.start_local_epoch]
        nc, nr = self.n_channels, self.n_records
        self._scaling = [[records[r]["channels"][c]["scaling"] for r in range(nr)]
                         for c in range(nc)]
        self.sample_rate = [[records[r]["tick_rate"] for r in range(nr)] for _ in range(nc)]
        self.n_samples = [[records[r]["channels"][c]["n_samples"] for r in range(nr)]
                          for c in range(nc)]
        self.units = [[s.unit for s in row] for row in self._scaling]
        self.comments = []
        for r, rec in enumerate(records):
            for com in rec["comments"]:
                self.comments.append({
                    "record": r, "channel": com["channel"], "tick": com["tick"],
                    "time_s": com["tick"] / rec["tick_rate"], "number": com["number"],
                    "text": com["text"]})
        self._check_chunk_layout()
        self._load_calculations(cf)

    def _load_calculations(self, cf):
        """Decide per channel: recorded input, reproducible Integral, or left out."""
        calcs = _read_calculations(cf)
        nc = self.n_channels
        self.calculation_on_unrecorded_channel = sorted(c + 1 for c in calcs if c >= nc)
        self.channel_kind = []
        self.integral_source = {}
        for c in range(nc):
            entry = calcs.get(c)
            if entry is None:
                self.channel_kind.append("recorded")
            elif (entry["kind"] == "integral" and entry["source"] < nc
                  and (entry["source"] == c or entry["source"] not in calcs)):
                # The source must be a recorded input, or this channel's own input.
                self.channel_kind.append("integral")
                self.integral_source[c] = entry["source"]
            else:
                self.channel_kind.append("calculation")
        for c, src in self.integral_source.items():
            for r in range(self.n_records):
                self.units[c][r] = self._scaling[src][r].unit + ".s"

    def is_exportable(self, channel):
        """False for a channel whose LabChart calculation cannot be reproduced."""
        return self.channel_kind[channel] != "calculation"

    @staticmethod
    def _read_version(cf):
        c = _Cursor(cf.read(cf.child(cf.root, "Version")), "Version")
        c.stream_magic()
        c.expect("IQ", 2, 0)
        first = c.string()
        second = c.string()
        return first if first == second else "%s / %s" % (first, second)

    def _check_chunk_layout(self):
        """Indexed chunks must tile Blocks_0 from the magic bytes to the end."""
        offsets = [ch["chunks"]["offset"] for rec in self._meta["records"]
                   for ch in rec["channels"]]
        allo = np.sort(np.concatenate(offsets)) if offsets else np.empty(0, dtype=np.uint64)
        self._chunk_ends = {}
        if allo.size == 0:
            if self._blocks.size != 4:
                raise AdichtFormatError("Blocks_0 holds data but the index lists no chunk.")
            return
        if allo[0] != 4:
            raise AdichtFormatError("First chunk does not start right after the magic bytes.")
        if np.any(np.diff(allo.astype(np.int64)) <= 20):
            raise AdichtFormatError("Chunk index holds overlapping chunks.")
        if allo[-1] >= self._blocks.size:
            raise AdichtFormatError("Chunk index points past the end of Blocks_0.")
        self._chunk_ends = dict(zip(allo.tolist(), allo[1:].tolist() + [self._blocks.size]))

    def _check_indices(self, channel, record):
        if not (0 <= channel < self.n_channels and 0 <= record < self.n_records):
            raise IndexError("channel %d, record %d out of range" % (channel, record))

    def _range(self, channel, record, first, stop):
        self._check_indices(channel, record)
        n = self.n_samples[channel][record]
        first = 0 if first is None else first
        stop = n if stop is None else stop
        if not 0 <= first <= stop <= n:
            raise IndexError("sample range %d to %d outside 0 to %d" % (first, stop, n))
        return first, stop

    def adc(self, channel, record, first=None, stop=None):
        """Raw ADC counts of the recorded input (int16), samples first to stop - 1."""
        first, stop = self._range(channel, record, first, stop)
        ch = self._meta["records"][record]["channels"][channel]
        out = np.empty(stop - first, dtype=np.int16)
        pos = 0          # index of the chunk's first sample in the record
        for e in ch["chunks"]:
            n = int(e["n"])
            lo_i, hi_i = max(first, pos), min(stop, pos + n)
            if lo_i < hi_i:
                off = int(e["offset"])
                values, used = _decode_chunk(self._blocks, off, n)
                if off + used != self._chunk_ends[off]:
                    raise AdichtFormatError(
                        "Chunk at %d ends at %d, the next chunk starts at %d."
                        % (off, off + used, self._chunk_ends[off])
                    )
                lo, hi = int(values.min()), int(values.max())
                # The index stores each chunk's extrema. They must match the decode.
                if lo != e["min"] or hi != e["max"]:
                    raise AdichtFormatError(
                        "Chunk at %d decodes to min %d, max %d but the index stores %g, %g."
                        % (off, lo, hi, e["min"], e["max"])
                    )
                if lo < -32768 or hi > 32767:
                    raise AdichtFormatError("Chunk at %d exceeds the 16 bit range." % off)
                out[lo_i - first : hi_i - first] = values[lo_i - pos : hi_i - pos]
            pos += n
            if pos >= stop:
                break
        return out

    def scaling(self, channel, record):
        self._check_indices(channel, record)
        s = self._scaling[channel][record]
        return {
            "unit": s.unit, "scale_units": s.scale_units, "scale_offset": s.scale_offset,
            "adc_scale": s.adc_scale, "base_unit": s.base_unit,
            "units_conversion": s.units_conversion, "unit_scale": s.unit_scale,
            "unit_offset": s.unit_offset, "calibration_text": list(s.calibration),
        }

    def input_data(self, channel, record, first=None, stop=None):
        """The recorded input in physical units (float64), whatever LabChart displays.

        value = (counts + scale_offset) * scale_units, evaluated in double
        precision. This is LabChart's own arithmetic: rounded to single
        precision it reproduces LabChart's MATLAB export bit for bit.
        """
        s = self._scaling[channel][record]
        counts = self.adc(channel, record, first, stop).astype(np.float64)
        return (counts + s.scale_offset) * s.scale_units

    def data(self, channel, record, first=None, stop=None):
        """The channel as LabChart exports it (float64).

        A recorded channel is its input. An Integral channel is recomputed.
        A channel with any other calculation raises AdichtFormatError.
        """
        first, stop = self._range(channel, record, first, stop)
        kind = self.channel_kind[channel]
        if kind == "recorded":
            return self.input_data(channel, record, first, stop)
        if kind == "integral":
            # LabChart: running sum, from the start of the record, of the
            # single precision source samples times the sample period, held in
            # double precision. Verified bit for bit after rounding to single.
            src = self.integral_source[channel]
            x = self.input_data(src, record, 0, stop).astype(np.float32).astype(np.float64)
            return np.cumsum(x * (1.0 / self.sample_rate[src][record]))[first:]
        raise AdichtFormatError(
            "Channel %d carries a LabChart channel calculation that this reader cannot "
            "reproduce. input_data() returns its recorded input." % (channel + 1)
        )

    def hardware_setup_xml(self, channel, record):
        self._check_indices(channel, record)
        return self._meta["records"][record]["channels"][channel]["setup_xml"]


def read_adicht(path):
    """Open a recording. Use as a context manager or call close()."""
    return AdichtFile(path)


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def _char_matrix(strings):
    """Rows of equal width, space padded, as LabChart writes text matrices."""
    width = max((len(s) for s in strings), default=0)
    for s in strings:
        try:
            s.encode("latin-1")
        except UnicodeEncodeError:
            raise AdichtFormatError(
                "Text %r cannot be stored in a MAT version 4 file (Latin-1 only)." % s
            )
    return np.array([s.ljust(width) for s in strings], dtype="U%d" % max(width, 1))


def plan_export(adi, channels=None, record=None, start=None, end=None):
    """Decide what an export holds. Returns a list of cells and a list of notes.

    channels: 0 based channel indices, default every channel that can be
    exported. record: 0 based record index, default every record. start and
    end: seconds from the start of that record, only with `record`.
    Each cell is (channel, record, first sample, stop sample) for the part
    that is written. Channels and records that are not written stay in the
    layout as empty cells, so row and column numbers keep their meaning.
    """
    nc, nr = adi.n_channels, adi.n_records
    if nc == 0 or nr == 0:
        raise AdichtFormatError("The recording holds no records.")
    notes = []
    left_out = [c for c in range(nc) if not adi.is_exportable(c)]
    if channels is None:
        channels = [c for c in range(nc) if c not in left_out]
    else:
        channels = sorted(set(channels))
        for c in channels:
            if not 0 <= c < nc:
                raise ValueError("Channel %d does not exist. The recording has channels 1 to %d."
                                 % (c + 1, nc))
            if c in left_out:
                raise AdichtFormatError(
                    "Channel %d (%s) carries a LabChart channel calculation that this reader "
                    "cannot reproduce. It cannot be exported." % (c + 1, adi.titles[c]))
    for c in left_out:
        notes.append("LEFT OUT: channel %d (%s) carries a LabChart channel calculation that "
                     "this reader cannot reproduce. Its row is empty (datastart -1)."
                     % (c + 1, adi.titles[c]))
    for c in sorted(adi.integral_source):
        notes.append("RECOMPUTED: channel %d (%s) is LabChart's Integral of channel %d (%s). "
                     "It is not stored in the file. It is recomputed with the arithmetic that "
                     "matched LabChart's exports bit for bit."
                     % (c + 1, adi.titles[c], adi.integral_source[c] + 1,
                        adi.titles[adi.integral_source[c]]))
    for c in adi.calculation_on_unrecorded_channel:
        notes.append("NOT IN THE FILE: LabChart shows a computed channel %d that has no "
                     "recorded input. It is not in this export." % c)
    if record is None:
        if start is not None or end is not None:
            raise ValueError("--start and --end need --record.")
        records = list(range(nr))
    else:
        if not 0 <= record < nr:
            raise ValueError("Record %d does not exist. The recording has records 1 to %d."
                             % (record + 1, nr))
        records = [record]
    cells = []
    for r in records:
        rate = adi.tick_rate[r]
        n = adi.n_ticks[r]
        first = 0 if start is None else int(round(start * rate))
        stop = n if end is None else int(round(end * rate))
        if not 0 <= first < stop <= n:
            raise ValueError(
                "The range %s to %s s is not inside record %d, which lasts %.6f s."
                % (start, end, r + 1, n / rate))
        if (first, stop) != (0, n) or len(records) != nr:
            notes.append("PARTIAL: record %d, samples %d to %d of %d (1 based, inclusive), "
                         "%.6f s to %.6f s from the start of the record. Other records are empty."
                         % (r + 1, first + 1, stop, n, first / rate, stop / rate))
        cells.extend((c, r, first, stop) for c in channels)
    if len(channels) != nc - len(left_out):
        notes.append("PARTIAL: channels %s only. Other channel rows are empty."
                     % ", ".join(str(c + 1) for c in channels))
    return cells, notes


def build_export(adi, channels=None, record=None, start=None, end=None):
    """Decode what plan_export selects. Returns (mat_dict, meta, notes).

    mat_dict follows LabChart's "Export MATLAB" layout: `data` is one row of
    single precision samples in physical units, stored record by record and
    channel by channel inside each record. datastart and dataend are 1 based
    inclusive indices into it, channels by records, -1 for an empty cell.
    """
    nc, nr = adi.n_channels, adi.n_records
    cells, notes = plan_export(adi, channels, record, start, end)
    total = sum(stop - first for _c, _r, first, stop in cells)
    # MAT version 4 is what LabChart itself writes. Its header stores the
    # number of samples as a signed 32 bit integer.
    if total > 2**31 - 1:
        raise AdichtFormatError(
            "The export would hold %d samples, more than a MAT version 4 file can hold. "
            "Export fewer channels or a part of a record." % total)
    data = np.empty(total, dtype=np.float32)
    datastart = np.full((nc, nr), -1.0)
    dataend = np.full((nc, nr), -1.0)
    unitmap = np.full((nc, nr), -1.0)
    unit_list = []
    written = {}
    pos = 0
    for c, r, first, stop in sorted(cells, key=lambda x: (x[1], x[0])):
        n = stop - first
        data[pos : pos + n] = adi.data(c, r, first, stop)
        datastart[c, r] = pos + 1
        dataend[c, r] = pos + n
        unit = adi.units[c][r]
        if unit not in unit_list:
            unit_list.append(unit)
        unitmap[c, r] = unit_list.index(unit) + 1
        written[(c, r)] = (first, stop)
        pos += n
    # Comment table columns: channel (-1 for all, else 1 based), record
    # (1 based), tick position from the start of the record (0 based), type
    # (1, a user comment), row of the text in comtext (1 based). Each distinct
    # text is stored once, in order of first use, as LabChart does. All
    # comments of the recording are kept, also in a partial export.
    texts = []
    com = np.zeros((len(adi.comments), 5))
    for i, cm in enumerate(adi.comments):
        if cm["text"] not in texts:
            texts.append(cm["text"])
        com[i] = (cm["channel"] + 1 if cm["channel"] >= 0 else -1, cm["record"] + 1,
                  cm["tick"], 1, texts.index(cm["text"]) + 1)
    mat = {
        "data": data.reshape(1, -1),
        "datastart": datastart,
        "dataend": dataend,
        "titles": _char_matrix(adi.titles),
        "unittext": _char_matrix(unit_list),
        "unittextmap": unitmap,
        "blocktimes": np.array([[t / 86400.0 + _MATLAB_DATENUM_OF_UNIX_EPOCH
                                 for t in adi.start_local_epoch]]),
        "tickrate": np.array([adi.tick_rate], dtype=float),
        "samplerate": np.array(adi.sample_rate, dtype=float).reshape(nc, nr),
        "firstsampleoffset": np.zeros((nc, nr)),
        "com": com,
    }
    if texts:
        mat["comtext"] = _char_matrix(texts)
    meta = {
        "source_file": adi.path.name,
        "software": adi.software,
        "notes": notes,
        "value_formula": "recorded channel: value = (adc_counts + scale_offset) * scale_units",
        "n_channels": nc,
        "n_records": nr,
        "titles": adi.titles,
        "all_titles_in_file": adi.all_titles,
        "records": [{
            "record": r + 1,
            "tick_rate_hz": adi.tick_rate[r],
            "n_ticks": adi.n_ticks[r],
            "duration_s": adi.n_ticks[r] / adi.tick_rate[r],
            "start_local": adi.start_local[r].isoformat(),
            "utc_offset_field_s": adi.utc_offset_s[r],
            "matlab_datenum_local": adi.start_local_epoch[r] / 86400.0
                                    + _MATLAB_DATENUM_OF_UNIX_EPOCH,
        } for r in range(nr)],
        "channels": [{
            "channel": c + 1,
            "title": adi.titles[c],
            "kind": adi.channel_kind[c],
            "integral_of_channel": adi.integral_source.get(c, -1) + 1 or None,
            "records": [{
                "record": r + 1,
                "unit": adi.units[c][r],
                "sample_rate_hz": adi.sample_rate[c][r],
                "n_samples_in_recording": adi.n_samples[c][r],
                "written": (c, r) in written,
                "first_sample_written": written[(c, r)][0] + 1 if (c, r) in written else None,
                "last_sample_written": written[(c, r)][1] if (c, r) in written else None,
                "start_s_from_record_start": (written[(c, r)][0] / adi.sample_rate[c][r]
                                              if (c, r) in written else None),
                "mat_datastart": int(datastart[c, r]),
                "mat_dataend": int(dataend[c, r]),
                "input_scaling": adi.scaling(c, r),
            } for r in range(nr)],
        } for c in range(nc)],
        "comments": [{
            "number": cm["number"], "record": cm["record"] + 1,
            "channel": cm["channel"] + 1 if cm["channel"] >= 0 else -1,
            "tick": cm["tick"], "time_s": cm["time_s"], "text": cm["text"],
        } for cm in adi.comments],
    }
    return mat, meta, notes


def write_outputs(adi, out_dir, channels=None, record=None, start=None, end=None):
    """Write <stem>.mat and <stem>.json into out_dir. Returns (paths, notes)."""
    import scipy.io

    out_dir = Path(out_dir)
    if out_dir.resolve() == adi.path.resolve().parent:
        raise ValueError("Refusing to write into the folder of the source recording.")
    mat, meta, notes = build_export(adi, channels, record, start, end)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = adi.path.stem
    paths = {k: out_dir / (stem + "." + k) for k in ("mat", "json")}
    scipy.io.savemat(str(paths["mat"]), mat, format="4")
    with open(paths["json"], "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return paths, notes


def _info(adi):
    lines = ["file: %s" % adi.path.name, "software: %s" % adi.software,
             "recorded channels: %d, records: %d" % (adi.n_channels, adi.n_records)]
    for r in range(adi.n_records):
        lines.append("record %d: %g Hz, %d ticks (%.3f s), start %s local time"
                     % (r + 1, adi.tick_rate[r], adi.n_ticks[r],
                        adi.n_ticks[r] / adi.tick_rate[r], adi.start_local[r].isoformat()))
    kinds = {"recorded": "recorded input", "calculation": "LabChart calculation, NOT REPRODUCED"}
    for c in range(adi.n_channels):
        kind = kinds.get(adi.channel_kind[c])
        if kind is None:
            kind = "LabChart Integral of channel %d, recomputed" % (adi.integral_source[c] + 1)
        units = [adi.units[c][r] if adi.is_exportable(c) else "?" for r in range(adi.n_records)]
        lines.append("  channel %d %-14r %s, unit per record: %s"
                     % (c + 1, adi.titles[c], kind, ", ".join(units)))
    for cm in adi.comments:
        lines.append("  comment %d: record %d, channel %s, %.4f s (tick %d): %r"
                     % (cm["number"], cm["record"] + 1,
                        "all" if cm["channel"] < 0 else cm["channel"] + 1,
                        cm["time_s"], cm["tick"], cm["text"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Convert a native LabChart recording (.adicht) to a .mat in LabChart's "
                    "export layout. The recording is opened read only.")
    ap.add_argument("adicht", type=Path, help="input recording")
    ap.add_argument("--out", type=Path, help="output directory (not the recording's folder)")
    ap.add_argument("--info", action="store_true", help="print the metadata and stop")
    ap.add_argument("--channels", help="channel numbers to export, 1 based, for example 1,3,5")
    ap.add_argument("--record", type=int, help="export this record only, 1 based")
    ap.add_argument("--start", type=float, help="seconds from the start of --record")
    ap.add_argument("--end", type=float, help="seconds from the start of --record")
    args = ap.parse_args(argv)
    if not args.info and args.out is None:
        ap.error("give --out <dir> or --info")
    try:
        channels = None
        if args.channels:
            channels = [int(x) - 1 for x in args.channels.split(",")]
        with AdichtFile(args.adicht) as adi:
            print(_info(adi))
            if not args.info:
                paths, notes = write_outputs(
                    adi, args.out, channels,
                    None if args.record is None else args.record - 1, args.start, args.end)
                for note in notes:
                    print(note)
                for path in paths.values():
                    print("wrote %s" % path)
    except AdichtFormatError as ex:
        print("REFUSED: %s" % ex, file=sys.stderr)
        return 2
    except (OSError, ValueError) as ex:
        print("ERROR: %s" % ex, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------
# FORMAT NOTES
# --------------------------------------------------------------------------
#
# All integers are little endian. Offsets are in bytes. "Verified" means
# checked against LabChart's own MATLAB exports of the same recordings (see
# section 6). "Not interpreted" means the reader checks the bytes against the
# values seen in verified files and refuses anything else.
#
# 1. Container: "ADInstruments Compound File"
#
#   0x000  UTF-16 text "ADInstruments Compound File", zero padded to 0x80.
#   0x080  14 u32 words. Word 1 is the header size (0x200). Word 6 is log2
#          of the block size (14, so 16384 bytes, in all but one file seen).
#          Word 7 is the capacity of the block table in entries and word 9
#          is its negative. Word 11 is the first block of the block table.
#          Words 0, 2 and 3 were 29, 0x20000 and 0x7FFFFFFF in every file
#          and are not interpreted.
#   0x0B8  Root directory entry, named "ADIRootStorage".
#   Block b (b >= 2) starts at 0x200 + (b - 2) * block_size.
#   Block table: one (next, prev) u32 pair per block, 0 meaning none. It
#   chains the blocks of every stream, its own blocks included. Entry 0 is a
#   marker (x, 0x80000000). The table sits in consecutive blocks near the
#   end of the file and unused blocks may follow it.
#   Directory entry, 160 bytes: 72 bytes UTF-16 name, u64 first block,
#   u64 size in bytes, u64 last block, 40 bytes not interpreted, u32 kind
#   (1 storage, 2 stream), 20 bytes not interpreted.
#   A storage is itself a stream: i32 count, i32 -count, then the entries.
#   Root storage: Settings (storage), Version, ChildDocuments (storage),
#   Map and Blocks_0. Only Version, Map and Blocks_0 are read here.
#
# 2. Map stream: all metadata and the index of the sample chunks
#
#   Serialised objects. A string is u32 0x80000001, i32 n, i32 -n, then n
#   UTF-16 units of which the last is NUL. An array is i32 n, i32 -n, then
#   n elements. A shared object is either defined in place (u32 id with the
#   top bit set, then its body) or referenced (u32 id).
#
#   Header: 4 magic bytes (aa aa 13 55 in files whose Version stream reads
#   "LabChart 6.1.2" twice, aa aa 15 55 in files whose Version stream names
#   LabChart 8.1.x), u32 2, u64 0, u16 length, ASCII "ADInstruments Chart
#   Data" with NUL, u16 3, u16 5 (older layout) or 7 (newer layout), u16
#   number of recorded channels of the document, u8 0.
#   Then records, each preceded by i32 index and u32 0. An index of -1 ends
#   the list.
#
#   Record: u32 version (1 older, 3 newer). u64 1 and u64 R: the tick
#   period is 1 / R seconds. f64 start time, seconds since 1970-01-01 in
#   the local time of the recording computer. f64 seconds to add to get UTC
#   (14400 in summer and 18000 in winter for these US Eastern recordings).
#   i32 pair, (0, 0) in direct recordings and (-X, +X) in files saved from
#   a selection (refused, see section 7). u32 0. u32 number of sample chunks
#   before this record. u64 number of ticks. Version 3 only: a string
#   (always empty), u32 0, u32 flags (0 or 0x30000000, not interpreted).
#   Array of tick chunks: u32 3, u32 ticks in the chunk (16000, less in the
#   last chunk, 0 for an empty trailing chunk), u16 0, u16 flags (0x8000 or
#   0xC000 on the first chunk, not interpreted), u16 0, u16 comment count,
#   then that many comments: u32 2, u32 0, u32 comment number, i16 channel
#   (-1 for all channels, else 0 based), u16 tick inside the chunk, string.
#   Array of channels.
#
#   Channel: bytes 01 00 00 1b 80, u32 version (2 older, 3 newer), u32
#   channel index, u32 1, a fixed 36 byte block, u64 sample rate (always
#   equal to R in verified files). Shared scaling object. Version 3 only:
#   u32 1. Shared data source object (the hardware setup as XML text).
#   u64 sample count. u32 (version 3: 0 or 0x80000000, not interpreted;
#   version 2: the 0 based record number). u32 longest chunk, u32 nominal
#   chunk length (16000). Array of chunk index entries, 36 bytes each:
#   u32 0x80020001, u64 0, u32 samples, u32 0, f32 minimum and f32 maximum
#   of the chunk in ADC counts, u64 offset of the chunk in Blocks_0.
#
#   Scaling object: u32 5, u32 2, f64 A (base units per ADC count, for
#   example 10 V / 32000), f64 ADC offset (always 0), f32 32767, f32
#   -32767, two u32 (display format, not interpreted), u16 1, string base
#   unit ("V"), f64 S and f64 O (units conversion: value = volts * S + O),
#   two u32 (display format, not interpreted), u16 0, u32 conversion switch
#   (0 off, 1 on), u16 flag bits (0, 0x4000 or 0x8000, not interpreted),
#   string unit name, four strings holding the two calibration points as
#   text, u8 0.
#
#   Trailer: array of channel title strings (one per channel slot of the
#   document, the recorded channels first), u64 total unpacked size, u64
#   size of the chunk data in Blocks_0, and in the newer layout one empty
#   string.
#
# 3. Blocks_0 stream: the samples
#
#   4 magic bytes 55 55 03 aa, then chunks back to back. The chunks of one
#   record are interleaved channel by channel.
#   Chunk: u32 L, u32 0, i32 -L, u16 0x0104, u16 0, u32 U (2 bytes per
#   sample), then L - 16 bytes of packed data. The next chunk starts at the
#   chunk offset + 4 + L.
#   Packed data, stage 1, run length coding: a control byte c below 0x80
#   copies the next c + 1 bytes. A control byte c of 0x80 or more repeats
#   the next byte 257 - c times.
#   Stage 2, bit planes: the unpacked bytes are 17 planes. Each plane is
#   ceil(n / 32) little endian 32 bit words. Bit i of a plane, counted from
#   the least significant bit, belongs to sample i. Plane 0 holds the sign
#   and planes 1 to 16 the magnitude, least significant plane first, of the
#   difference to the previous sample. The first difference is taken from
#   zero, so every chunk decodes on its own.
#   The result is signed 16 bit ADC counts.
#
# 4. Physical units
#
#   scale_units  = A * S          (S = 1 when the units conversion is off)
#   scale_offset = O / (A * S)    (0 when the units conversion is off)
#   value        = (counts + scale_offset) * scale_units, in double precision
#   This is the convention of LabChart's 16 bit MATLAB export (scaleunits,
#   scaleoffset). Rounded to single precision it reproduces LabChart's
#   floating point MATLAB export bit for bit.
#
# 5. Channel calculations: Settings/SettingsRegistrar
#
#   LabChart stores only recorded inputs. A channel that LabChart shows as
#   a channel calculation is recomputed on display and on export, and the
#   file holds the raw input of that channel instead.
#   The SettingsRegistrar stream holds one entry per channel that carries
#   a calculation: bytes 23 00 0c 80, u32 0, u32 channel index (0 based),
#   u16 0, u16 not interpreted, then the calculation's settings.
#   Integral: 16 byte class id 10 4b 96 23 32 90 d4 11 81 11 00 a0 c9 e2
#   64 33, u32 object id with the top bit set, u16 1, u16 version (6 to
#   11), u32 not interpreted, u32 source channel index (0 based), i32 -1,
#   f64 1.0, f64 -1.0, u32 2 or 3, u8 1, string "sta<TAB>off".
#   LabChart's arithmetic for this Integral, found by matching exports:
#     value[i] = sum over j <= i of float64(float32(source[j])) * (1 / rate)
#   The sum restarts at every record and is held in double precision. The
#   unit is the source unit followed by ".s".
#   The reader recomputes an Integral only when every field above has a
#   value seen in a verified recording and the source is a recorded input
#   (or the channel's own input). Every other tagged channel is left out.
#   The rest of the stream is not decoded. The tag is found by scanning for
#   its 8 bytes.
#
# 6. What was verified, and against what
#
#   LabChart's own MATLAB exports (MAT version 4, single precision) of the
#   same recordings: 72 pairs covering 68 recordings made between 2016
#   and 2026 at 20, 40 and 100 kHz. Their Version streams read "LabChart
#   6.1.2" alone, or name LabChart 8.1.11 to 8.1.30.
#   Channel count, record count, samples per channel and record, titles,
#   units and comments (record, channel, tick, text) agreed in all pairs.
#   Record start times agreed to the last bit of the MATLAB date number.
#   Recorded channels: 844 channel records, 4,434,174,000 samples, all
#   equal to LabChart's export bit for bit in single precision.
#   Integral channels: 285 channel records, 1,284,461,000 samples, all
#   equal bit for bit.
#   11 channel records were left out by the reader. 6 hold an Integral
#   whose settings lack the class id (2 older files). 5 are a recorded
#   channel of one file in which the tag bytes also occur without a
#   calculation, so the scan errs on the side of leaving a channel out.
#   No channel record differed from its export.
#   No recording with a calculation other than Integral had an export.
#   In addition every chunk's decoded minimum and maximum must equal the
#   values stored in its index entry. This check runs on every read. It is
#   not a checksum: a damaged chunk whose extrema survive would pass.
#
# 7. What the reader refuses or leaves out
#
#   Left out (empty row, datastart -1): a channel with a calculation that
#   is not a verified Integral. If a calculation were stored without the
#   tag of section 5, the reader would return the raw input unnoticed. No
#   such case occurred in the verified pairs.
#   Refused: files saved from a selection of a longer recording (start
#   offset field not zero) and files from which records were deleted
#   (Blocks_0 size disagrees with the Map). No LabChart export of such a
#   file was at hand.
#   Refused: channels sampled below the tick rate, channels that do not
#   span their whole record, records that hold fewer channels than the
#   document, and recordings imported from other systems (seen in two
#   polysomnography imports). Not decoded.
#   Refused: Map versions, record flags, display format words, units flag
#   bits and ADC ranges other than those in the _VERIFIED_ tables above.
#   Recordings with 3, 4 and 5 recorded channels were verified. Other
#   channel counts use the same layout and are accepted.
#   Refused: any other deviation from the layout above.
#
# 8. Not decoded, not written
#
#   Everything else under Settings and ChildDocuments (display settings,
#   macros, device descriptions). LabChart's rangemin and rangemax export
#   variables are display ranges and are not written. Tick rates are
#   written as R exactly. LabChart's export writes 1 / (1 / R), which is
#   99999.99999999999 for R = 100000.
#   A partial export keeps LabChart's full layout: channels and records
#   that are not written are empty cells (datastart -1, unittextmap -1).
#   No LabChart partial export was at hand to compare that layout with.
