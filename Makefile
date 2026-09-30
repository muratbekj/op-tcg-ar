# Uses the full Xcode toolchain even when xcode-select points at the Command Line Tools.
export DEVELOPER_DIR ?= /Applications/Xcode.app/Contents/Developer

PROJECT := apps/ios/OnePieceAR.xcodeproj
PACKAGE := apps/ios/Packages/OnePieceKit

.PHONY: test build open ml-setup ml-test eval

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
