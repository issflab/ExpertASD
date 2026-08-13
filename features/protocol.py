"""Protocol-file parsing.

Two layouts are supported and auto-detected:

  csv       header row, comma- or tab-separated, columns found by name
  asvspoof  whitespace-separated, ASVspoof2019/2021 style, e.g.
            `LA_0079 LA_T_1138215 - A01 spoof`

Labels are normalised to exactly `real` / `fake` at parse time so nothing
downstream has to know that a corpus said "bonafide" or "1". Anything a corpus
uses that is not in LABEL_ALIASES is an error, not a silent guess -- a
mislabelled point would quietly poison every plot built from it.
"""
from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .config import ExtractionConfig, ProtocolConfig

logger = logging.getLogger(__name__)

REAL = "real"
FAKE = "fake"

LABEL_ALIASES = {
    "bonafide": REAL,
    "bona-fide": REAL,
    "bona_fide": REAL,
    "genuine": REAL,
    "real": REAL,
    "human": REAL,
    "0": REAL,
    "spoof": FAKE,
    "spoofed": FAKE,
    "fake": FAKE,
    "synthetic": FAKE,
    "1": FAKE,
}

#: Column-name candidates per role, tried in order, matched case-insensitively.
CSV_COLUMN_CANDIDATES = {
    "filename": ["filename", "file", "file_name", "path", "filepath", "audio",
                 "audio_file", "audio_path", "wav", "utt", "utt_id", "utterance", "name", "id"],
    "label": ["label", "key", "class", "target", "ground_truth", "attack_type_label"],
    "system": ["system", "system_id", "attack", "attack_id", "attack_type",
               "tts_system", "source", "spoof_type", "method"],
    "speaker": ["speaker", "speaker_id", "spk", "spk_id", "subject"],
}

_ASVSPOOF_SYSTEM_RE = re.compile(r"^[A-Z]\d{2}$")  # A01..A19, etc.
_NULL_TOKENS = {"-", "--", "", "none", "null", "na", "n/a", "nan"}


class ProtocolError(ValueError):
    """Raised when a protocol file cannot be parsed unambiguously."""


@dataclass
class ProtocolRecord:
    """One protocol line, with `path` filled in once the audio is resolved."""

    filename: str
    label: str
    system: Optional[str] = None
    speaker: Optional[str] = None
    path: Optional[Path] = None


def _clean(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    token = token.strip()
    return None if token.lower() in _NULL_TOKENS else token


def normalize_label(raw: str, line_no: int) -> str:
    """Map a corpus label onto `real`/`fake`, or fail loudly."""
    key = str(raw).strip().lower()
    if key not in LABEL_ALIASES:
        raise ProtocolError(
            f"line {line_no}: unrecognised label '{raw}'. "
            f"Known labels: {', '.join(sorted(LABEL_ALIASES))}"
        )
    return LABEL_ALIASES[key]


# -- format detection ------------------------------------------------------


def _sniff_delimiter(line: str) -> str:
    return "\t" if line.count("\t") > line.count(",") else ","


def detect_format(path: Path, cfg: ProtocolConfig) -> str:
    """Return `csv` or `asvspoof`, honouring an explicit `protocol.format`."""
    if cfg.format != "auto":
        return cfg.format

    first = ""
    with path.open() as fh:
        for line in fh:
            if line.strip():
                first = line.strip()
                break
    if not first:
        raise ProtocolError(f"protocol file is empty: {path}")

    delim = _sniff_delimiter(first)
    if delim in first:
        header = [t.strip().lower() for t in first.split(delim)]
        known = {name for names in CSV_COLUMN_CANDIDATES.values() for name in names}
        if len(header) > 1 and any(h in known for h in header):
            return "csv"
    return "asvspoof"


# -- csv -------------------------------------------------------------------


def _resolve_csv_columns(header: Sequence[str], cfg: ProtocolConfig) -> Dict[str, Optional[str]]:
    lookup = {h.strip().lower(): h for h in header}
    resolved: Dict[str, Optional[str]] = {}
    for role, candidates in CSV_COLUMN_CANDIDATES.items():
        override = cfg.columns.get(role)
        if override is not None:
            if str(override) not in header:
                raise ProtocolError(
                    f"protocol.columns.{role} = '{override}' is not a column in the "
                    f"protocol header: {list(header)}"
                )
            resolved[role] = str(override)
            continue
        resolved[role] = next((lookup[c] for c in candidates if c in lookup), None)

    for required in ("filename", "label"):
        if resolved[required] is None:
            raise ProtocolError(
                f"could not find a '{required}' column in the protocol header "
                f"{list(header)}; set protocol.columns.{required} in the config"
            )
    return resolved


def _normalize_delimiter(delim: str) -> str:
    """Coerce a configured delimiter to the single character csv needs.

    YAML single-quoted `'\\t'` is the two literal characters backslash-t, not a
    tab -- a common config trap. Decode such escapes so both quote styles work,
    then insist on exactly one character.
    """
    if len(delim) > 1:
        try:
            decoded = delim.encode().decode("unicode_escape")
        except UnicodeDecodeError:
            decoded = delim
        if len(decoded) == 1:
            return decoded
        raise ProtocolError(
            f"protocol.delimiter must be a single character, got {delim!r}. "
            'For a tab use double quotes ("\\t") in YAML, not single quotes.'
        )
    return delim


def _parse_csv(path: Path, cfg: ProtocolConfig) -> List[ProtocolRecord]:
    with path.open(newline="") as fh:
        sample = fh.readline()
        fh.seek(0)
        delim = _normalize_delimiter(cfg.delimiter) if cfg.delimiter else _sniff_delimiter(sample)
        reader = csv.DictReader(fh, delimiter=delim)
        if reader.fieldnames is None:
            raise ProtocolError(f"protocol file has no header row: {path}")
        cols = _resolve_csv_columns(reader.fieldnames, cfg)
        records = []
        for line_no, row in enumerate(reader, start=2):
            filename = _clean(row.get(cols["filename"]))
            if filename is None:
                raise ProtocolError(f"line {line_no}: empty filename")
            raw_label = row.get(cols["label"])
            if raw_label is None or not str(raw_label).strip():
                raise ProtocolError(f"line {line_no}: empty label")
            records.append(
                ProtocolRecord(
                    filename=filename,
                    label=normalize_label(raw_label, line_no),
                    system=_clean(row.get(cols["system"])) if cols["system"] else None,
                    speaker=_clean(row.get(cols["speaker"])) if cols["speaker"] else None,
                )
            )
    return records


# -- asvspoof --------------------------------------------------------------


def _index_override(cfg: ProtocolConfig, role: str, line_no: int) -> Optional[int]:
    value = cfg.columns.get(role)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ProtocolError(
            f"protocol.columns.{role} must be a 0-based integer index for "
            f"whitespace-separated protocols, got '{value}'"
        ) from None


def _parse_asvspoof_line(tokens: List[str], cfg: ProtocolConfig, line_no: int) -> ProtocolRecord:
    label_idx = _index_override(cfg, "label", line_no)
    if label_idx is None:
        # Search from the right: 2021 protocols append trim/subset columns after
        # the label, but no other column is ever a label word.
        label_idx = next(
            (i for i in range(len(tokens) - 1, -1, -1)
             if tokens[i].strip().lower() in LABEL_ALIASES),
            None,
        )
    if label_idx is None or not 0 <= label_idx < len(tokens):
        raise ProtocolError(
            f"line {line_no}: no label found among tokens {tokens}; "
            "set protocol.columns.label to the 0-based column index"
        )
    label = normalize_label(tokens[label_idx], line_no)

    file_idx = _index_override(cfg, "filename", line_no)
    if file_idx is None:
        # ASVspoof convention is `<speaker> <file> ...`; a bare two-column
        # protocol is `<file> <label>`.
        file_idx = 1 if len(tokens) >= 3 else 0
        if file_idx == label_idx:
            file_idx = 0
    if not 0 <= file_idx < len(tokens):
        raise ProtocolError(f"line {line_no}: filename index {file_idx} out of range for {tokens}")
    filename = _clean(tokens[file_idx])
    if filename is None:
        raise ProtocolError(f"line {line_no}: empty filename")

    speaker_idx = _index_override(cfg, "speaker", line_no)
    speaker = _clean(tokens[speaker_idx]) if speaker_idx is not None else (
        _clean(tokens[0]) if file_idx != 0 else None
    )

    system_idx = _index_override(cfg, "system", line_no)
    if system_idx is not None:
        system = _clean(tokens[system_idx])
    else:
        reserved = {file_idx, label_idx, 0 if speaker is not None else -1}
        system = next(
            (_clean(tokens[i]) for i in range(len(tokens))
             if i not in reserved and _ASVSPOOF_SYSTEM_RE.match(tokens[i].strip())),
            None,
        )
        if system is None and label_idx - 1 not in reserved and label_idx - 1 >= 0:
            system = _clean(tokens[label_idx - 1])

    return ProtocolRecord(filename=filename, label=label, system=system, speaker=speaker)


def _parse_asvspoof(path: Path, cfg: ProtocolConfig) -> List[ProtocolRecord]:
    records = []
    with path.open() as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            records.append(_parse_asvspoof_line(line.split(), cfg, line_no))
    return records


# -- public api ------------------------------------------------------------


def load_protocol(cfg: Union[ExtractionConfig, Tuple[Path, ProtocolConfig]]) -> List[ProtocolRecord]:
    """Parse the protocol into records, without touching the audio directory."""
    if isinstance(cfg, ExtractionConfig):
        path, pcfg = cfg.protocol_path, cfg.protocol
    else:
        path, pcfg = cfg
    path = Path(path)
    if not path.exists():
        raise ProtocolError(f"protocol file not found: {path}")

    fmt = detect_format(path, pcfg)
    records = _parse_csv(path, pcfg) if fmt == "csv" else _parse_asvspoof(path, pcfg)
    if not records:
        raise ProtocolError(f"protocol file contained no usable rows: {path}")

    n_real = sum(1 for r in records if r.label == REAL)
    logger.info(
        "parsed %d records from %s (format=%s, real=%d, fake=%d, systems=%d)",
        len(records), path, fmt, n_real, len(records) - n_real,
        len({r.system for r in records if r.system}),
    )
    return records


def resolve_audio(
    records: Sequence[ProtocolRecord],
    audio_dir: Path,
    extensions: Sequence[str],
) -> Tuple[List[ProtocolRecord], List[ProtocolRecord]]:
    """Attach an on-disk path to each record.

    Returns (found, missing). Every missing file is collected rather than
    raising on the first one -- a 10k-file protocol with a handful of gaps
    should still produce embeddings for the rest.
    """
    found, missing = [], []
    for record in records:
        candidate = audio_dir / record.filename
        resolved = candidate if candidate.is_file() else None
        if resolved is None and candidate.suffix.lower() not in {e.lower() for e in extensions}:
            for ext in extensions:
                with_ext = candidate.with_name(candidate.name + ext)
                if with_ext.is_file():
                    resolved = with_ext
                    break
        if resolved is None:
            missing.append(record)
        else:
            record.path = resolved
            found.append(record)

    if missing:
        logger.warning(
            "%d/%d protocol entries have no audio under %s (first: %s)",
            len(missing), len(records), audio_dir, missing[0].filename,
        )
    return found, missing
