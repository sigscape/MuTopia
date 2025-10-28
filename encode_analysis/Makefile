
# Enable secondary expansion for multi-wildcard rules
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -e -u -c

.SECONDEXPANSION:
# Prevent Make from deleting config files when downstream targets fail
.PRECIOUS: configs/features/%.yaml staged/%.train.nc staged/%.test.nc gtensors/%.nc studies/%/study.journal gtensors/%.features

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
UBERON_IDS_Lung-All := "UBERON:0002048" "UBERON:0002168" "UBERON:0002167" "UBERON:0008953" "UBERON:0008952" "UBERON:0002170" "UBERON:0002171"
REPLISEQ_Lung-All := "EFO:0001196"

UBERON_IDS_Ovary-AdenoCA := "UBERON:0000992"
REPLISEQ_Ovary-AdenoCA := "EFO:0001203"

UBERON_IDS_Jake-Breast := "UBERON:0008367" "UBERON:0000310" "CL:0002327"
REPLISEQ_Jake-Breast := "EFO:0001203"

UBERON_IDS_Prost-AdenoCA := "UBERON:0002367" "CL:0002231"
REPLISEQ_Prost-AdenoCA := "EFO:0001203"

UBERON_IDS_Myeloid-All := "CL:0001059" "CL:0000837" "CL:2000001" "EFO:0002784"
REPLISEQ_Myeloid-All := "EFO:0002784"

UBERON_IDS_Panc-AdenoCA := "UBERON:0001264" "UBERON:0001150"
REPLISEQ_Panc-AdenoCA := "EFO:0001187"

UBERON_IDS_Kidney-All := "UBERON:0002113" "UBERON:0004538" "UBERON:0004539" "UBERON:0001225" "CL:0002518" "CL:1000510" "CL:1000892"
REPLISEQ_Kidney-All := "EFO:0001187"

UBERON_IDS_Panc-Endocrine := "UBERON:0000016" "CL:0002351" "UBERON:0001264"
REPLISEQ_Panc-Endocrine := "EFO:0001187"

UBERON_IDS_Stomach-AdenoCA := "UBERON:0000945"
REPLISEQ_Stomach-AdenoCA := "EFO:0001187"

UBERON_IDS_Skin-Melanoma := "CL:1000458" "UBERON:0002097" "UBERON:0001003" "UBERON:0004264" "UBERON:0036149"
REPLISEQ_Skin-Melanoma := "EFO:0001196"

UBERON_IDS_Liver-HCC := "UBERON:0002107" "UBERON:0001115" "UBERON:0001114"
REPLISEQ_Liver-HCC := "EFO:0001187"

UBERON_IDS_Colon-AdenoCA := "UBERON:0000317" "UBERON:0004992" "UBERON:0001159" "UBERON:0001157" "UBERON:0008971"
REPLISEQ_Colon-AdenoCA := "EFO:0001187"

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
	gtensor compose base_config.yaml configs/features/$*.yaml -o gtensors/$* -w 5 && \
		gtensor feature ls gtensors/$*.nc > gtensors/$*.features

gtensors/%.samples gtensors/%.nc: gtensors/%.features
	@mkdir -p $(dir $@)
	gtensor compose base_config.yaml configs/features/$*.yaml configs/samples/$*.yaml -o gtensors/$* -w 15 && \
		gtensor info gtensors/$*.nc > gtensors/$*.samples

staged/%.done staged/%.train.nc staged/%.test.nc: gtensors/%.samples gtensors/%.nc
	@mkdir -p staged
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