
# Enable secondary expansion for multi-wildcard rules
.SECONDEXPANSION:

# Prevent Make from deleting config files when downstream targets fail
.PRECIOUS: configs/features/%.yaml staged/%.train.nc staged/%.test.nc gtensors/%.nc studies/%/study.journal

# Mark temp files as intermediate - they'll be deleted on failure and after successful completion
.INTERMEDIATE: gtensor__tempfiles/processed/%/

# Phony targets for convenience - allows "make lung" instead of "make data/gtensors/lung.done"
.PHONY: run/% uberon_terms/%

configs/repliseq.yaml: download_encode.py
	python download_encode.py --genome hg19 --assays "Repli-seq" --no-groupby > $@

configs/functional_data.yaml: download_encode.py
	python download_encode.py > $@

configs/experiments.yaml: configs/repliseq.yaml configs/functional_data.yaml
	cat $^ > $@

uberon_terms/%: configs/experiments.yaml
	python get_uberon_terms.py $< $*


# Define tissue-specific UBERON IDs and repliseq terms
UBERON_IDS_Lung-Adeno := "UBERON:0002048" "UBERON:0002168" "UBERON:0002167" "UBERON:0008953" "UBERON:0008952" "UBERON:0002170" "UBERON:0002171"
REPLISEQ_Lung-Adeno := "EFO:0001196"

UBERON_IDS_AML := "CL:0000837" "CL:2000001" 
REPLISEQ_AML := "EFO:0002783"

UBERON_IDS_Ovary-Adeno := "UBERON:0000992"
REPLISEQ_Ovary-Adeno := "EFO:0001203"

UBERON_IDS_Breast := "UBERON:0008367" "UBERON:0000310"
REPLISEQ_Breast := "EFO:0001203"

UBERON_IDS_Skin-Melanoma := "CL:1000458" "UBERON:0002097" "UBERON:0001003" "UBERON:0004264" "UBERON:0036149"
REPLISEQ_Skin-Melanoma := "EFO:0001196"

# Pattern rule that uses tissue-specific variables
configs/features/%.yaml: configs/experiments.yaml base_config.yaml write_config.py
	@mkdir -p $(dir $@)
	python write_config.py \
		base_config.yaml \
		$< \
		--name $* \
		-ids $(UBERON_IDS_$*) \
		-repliseq $(REPLISEQ_$*) \
		-o $@

gtensors/%.features gtensors/%.nc gtensor__tempfiles/processed/%/: configs/features/%.yaml
	@mkdir -p $(dir $@)
	gtensor compose base_config.yaml configs/features/$*.yaml -o gtensors/$* -w 5 && touch gtensors/$*.features

gtensors/%.samples gtensors/%.nc: gtensors/%.features
	gtensor compose base_config.yaml configs/features/$*.yaml configs/samples/$*.yaml -o gtensors/$* -w 5 && touch gtensors/$*.samples

staged/%.done staged/%.train.nc staged/%.test.nc: gtensors/%.samples gtensors/%.nc
	mkdir -p staged
	gtensor split gtensors/$*.nc -o staged/$*. "chr2" && touch $@

# Optional arguments that can be passed via command line
STUDY_ARGS_Lung-Adeno := -min 10 -max 20
STUDY_ARGS_Ovary-Adeno := -min 5 -max 15 -e -lsub 0.25

studies/%/study.journal: staged/$$(firstword $$(subst /, ,$$*)).train.nc staged/$$(firstword $$(subst /, ,$$*)).test.nc
	@mkdir -p $(dir $@)
	topo-model study create $@ -ds $^ $(STUDY_ARGS_$(firstword $(subst /, ,$*)))

run/%/: studies/%/study.journal
	topo-model study run $< -@ 5 --lazy

summaries/%.summary.txt: studies/%/study.journal
	@mkdir -p $(dir $@)
	topo-model study summary $< -o $@

analyses/%.analysis.ipynb: studies/%/study.journal
	@mkdir -p $(dir $@)