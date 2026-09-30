# Uses the full Xcode toolchain even when xcode-select points at the Command Line Tools.
export DEVELOPER_DIR ?= /Applications/Xcode.app/Contents/Developer

PROJECT := apps/ios/OnePieceAR.xcodeproj
PACKAGE := apps/ios/Packages/OnePieceKit

.PHONY: test build open ml-setup ml-test eval status freeze-test ship-baseline pull-model import-scans

## Unit tests for models, catalog, recognition math, and battle rules (no device needed). Serial: Vision tests deadlock in parallel.
test:
	swift test --no-parallel --package-path $(PACKAGE)

## Compile the app for a generic iPhone without signing (CI-style sanity check).
build:
	xcodebuild -project $(PROJECT) -scheme OnePieceAR -destination 'generic/platform=iOS' \
		CODE_SIGNING_ALLOWED=NO -quiet build

open:
	open $(PROJECT)

## Python lab environment (uv).
ml-setup:
	cd ml && uv sync

ml-test:
	cd ml && uv run pytest -q

## Recognition eval through the device pipeline; report in ml/runs/, history in ml/results/results.csv.
eval:
	cd ml && uv run scripts/evaluate.py --name $(or $(NAME),manual)

## Mac mini: import scans copied into ~/oplab-inbox (SMB) and archive the originals.
import-scans:
	cd ml && uv run scripts/prepare_dataset.py import-inbox

## Scan labels, train/test split, and progress toward the next frozen test set.
status:
	cd ml && uv run scripts/prepare_dataset.py status

## Freeze the next real-scan test set (needs ≥200 labeled test scans across ≥30 printings).
freeze-test:
	cd ml && uv run scripts/prepare_dataset.py freeze-test

## Ship the current Vision feature-print index as the baseline (v0) into ml/shipped/.
ship-baseline:
	cd ml && uv run scripts/ship.py baseline --name $(or $(NAME),v0)

## MacBook: fetch ml/shipped/ from the Mac mini (ml/remote.env) and install it for the next app build.
pull-model:
	cd ml && uv run scripts/remote.py pull-model
