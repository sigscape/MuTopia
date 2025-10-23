
# Enable secondary expansion for multi-wildcard rules
.SECONDEXPANSION:

# Phony targets for convenience - allows "make lung" instead of "make data/gtensors/lung.done"
.PHONY: lung ovary lung.features lung.samples ovary.features ovary.samples

data/repliseq.yaml:
	python download_encode.py --genome hg19 --assays "Repli-seq" --no-groupby > $@

data/functional_data.yaml:
	python download_encode.py > $@

data/experiments.yaml: data/repliseq.yaml data/functional_data.yaml
	cat $^ > $@

configs/features/lung.yaml: data/experiments.yaml
	python write_config.py \
		$< \
		-ids "UBERON:0002048" "UBERON:0002168" "UBERON:0002167" "UBERON:0008953" "UBERON:0008952" "UBERON:0002170" "UBERON:0002171" \
		-repliseq "EFO:0001196" \
		> $@

configs/features/ovary.yaml: data/experiments.yaml
	python write_config.py \
		$< \
		-ids "UBERON:0001261" "UBERON:0001736" "UBERON:0004493" "UBERON:0004494" "UBERON:0004495" "UBERON:0004496" \
		-repliseq "EFO:0001196" \
		> $@

gtensors/%.features:
	gtensor compose configs/features/$*.yaml --name $* -w 1 && touch $@

gtensors/%.samples: gtensors/%.features
	gtensor compose configs/samples/$*.yaml --name $* -w 1 && touch $@

staged/%.train.nc staged/%.test.nc: gtensors/%.samples
	gtensor split $< -o staged/$*

# Optional arguments that can be passed via command line
# Usage: make studies/lung/01/journal.journal STUDY_ARGS="--epochs 100 --lr 0.001"
STUDY_ARGS ?=

# Generic fallback for other tissues
studies/%/journal.journal: staged/$$(firstword $$(subst /, ,$$*)).train.nc staged/$$(firstword $$(subst /, ,$$*)).test.nc
	@mkdir -p $(dir $@)
	topo-model study create $@ -ds $^ $(STUDY_ARGS)