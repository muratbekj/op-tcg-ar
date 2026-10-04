# Uses the full Xcode toolchain when it's installed, even if xcode-select points at the Command Line
# Tools. Without Xcode (e.g. the Mac mini) the Command Line Tools build the ML lab's Swift CLI; the
# app's `test`/`build` targets still need Xcode.
XCODE_DEVELOPER := /Applications/Xcode.app/Contents/Developer
ifneq ($(wildcard $(XCODE_DEVELOPER)),)
export DEVELOPER_DIR ?= $(XCODE_DEVELOPER)
endif

PROJECT := apps/ios/OnePieceAR.xcodeproj
PACKAGE := apps/ios/Packages/OnePieceKit

.PHONY: test build open ml-setup ml-test eval status freeze-test ship-baseline pull-model import-scans mini-doctor ml-setup-train train export train-remote

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

## Evaluate model version NAME (v0 = Vision feature print) on the latest frozen test set; writes
## ml/models/NAME/{metrics.json, MODEL_CARD.md}. DIAG=1 uses the synthetic/photo manifest (not shippable).
eval:
	@test -n "$(NAME)" || { echo "usage: make eval NAME=v1 [DIAG=1]"; exit 2; }
	cd ml && uv run scripts/registry.py eval $(NAME) $(if $(DIAG),--diagnostic,)

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

## Python lab with training extras (torch, coremltools): Mac mini.
ml-setup-train:
	cd ml && uv sync --extra train

## Mac mini: check the one-time two-Mac setup (Swift toolchain, uv, SMB inbox, Remote Login, index).
mini-doctor:
	cd ml && uv run scripts/remote.py doctor

## Mac mini: fine-tune an embedder, keeping the Mac awake. NAME=v1 required; ARGS passes extra options.
train:
	@test -n "$(NAME)" || { echo "usage: make train NAME=v1 [ARGS='--epochs 10']"; exit 2; }
	cd ml && caffeinate -i uv run scripts/train_embedding.py --name $(NAME) $(ARGS)

## Mac mini: export a trained run to Core ML (ml/models/NAME/CardEmbedder.mlpackage).
export:
	@test -n "$(NAME)" || { echo "usage: make export NAME=v1"; exit 2; }
	cd ml && uv run scripts/export_coreml.py --name $(NAME)

## MacBook: start `make train NAME=…` on the Mac mini (ml/remote.env) inside tmux.
train-remote:
	@test -n "$(NAME)" || { echo "usage: make train-remote NAME=v1 [ARGS='--epochs 10']"; exit 2; }
	cd ml && uv run scripts/remote.py train-remote "$(NAME)" --args="$(ARGS)"
