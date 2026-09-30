"""Repo locations shared by every script."""

from pathlib import Path

ML = Path(__file__).resolve().parent.parent
REPO = ML.parent

DATA_CARDS = REPO / "data" / "cards"
ROSTER = DATA_CARDS / "roster.json"
ART = DATA_CARDS / "art"  # API art, gitignored
INDEX = DATA_CARDS / "printings.f32"
INDEX_META = DATA_CARDS / "printings.meta.json"
FULL_CATALOG = DATA_CARDS / "catalog.json"  # every printing in the API (tracked; bundled into the app)

APP_RESOURCES = REPO / "apps" / "ios" / "OnePieceAR" / "Resources"
APP_CARDS = APP_RESOURCES / "Cards"  # art bundled in the app (API art or your own clean scans)
APP_MODELS = APP_RESOURCES / "Models"
SWIFT_PACKAGE = REPO / "apps" / "ios" / "Packages" / "OnePieceKit"

DATASETS = ML / "datasets"
RAW = DATASETS / "raw"
API_CACHE = RAW / "optcg"
SCANS = RAW / "scans"  # exported from the device's Documents/Scans
PHOTOS = RAW / "photos"  # your own photos: photos/<printingId>/[<condition>/]*.jpg
SYNTH = RAW / "synth"
NEGATIVES = RAW / "negatives"
PROCESSED = DATASETS / "processed"
REFERENCES = DATASETS / "references"
TEST = DATASETS / "test"
TEST_MANIFEST = TEST / "manifest.json"
TESTSETS = ML / "testsets"  # frozen real-scan test sets, tracked in git

MODELS = ML / "models"
RUNS = ML / "runs"
RESULTS = ML / "results" / "results.csv"  # tracked in git: the history of every evaluation
SHIPPED = ML / "shipped"  # what the app should bundle; rsynced to the MacBook (gitignored)
REMOTE_ENV = ML / "remote.env"  # MacBook only: MINI_HOST, MINI_REPO (gitignored)
INBOX = Path.home() / "oplab-inbox"  # Mac mini: the SMB share the iPhone copies Scans into
