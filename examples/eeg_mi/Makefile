.PHONY: all train benchmark clean help

all: train benchmark

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all        Run the full pipeline (train -> benchmark)"
	@echo "  train      Train the EEG-SNN model (dataset auto-downloaded)"
	@echo "  benchmark  Run NeuroBench evaluation on the val set"
	@echo "  clean      Remove all downloaded dataset folder"
	@echo "  help       Show this message"

train:
	python train.py

benchmark: ./model_data/best_model.pt
	python benchmark.py

clean:
	rm -rf ../../data/