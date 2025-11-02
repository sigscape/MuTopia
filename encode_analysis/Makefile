
# Enable secondary expansion for multi-wildcard rules
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -e -u -c

.SECONDEXPANSION:
# Prevent Make from deleting config files when downstream targets fail
.PRECIOUS: configs/features/%.yaml staged/%.train.nc staged/%.test.nc gtensors/%.nc gtensors/%.features

# Phony targets for convenience - allows "make lung" instead of "make data/gtensors/lung.done"
.PHONY: run/% uberon_terms/% summarize/%/ stage.all

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

UBERON_IDS_Kidney-All := "UBERON:0002113" "UBERON:0004538" "UBERON:0004539" "UBERON:0001225" "CL:0002518" "CL:1000510" "CL:1000892" "UBERON:0001255"
REPLISEQ_Kidney-All := "EFO:0001187"

UBERON_IDS_Panc-Endocrine := "UBERON:0000016" "CL:0002351" "UBERON:0001264"
REPLISEQ_Panc-Endocrine := "EFO:0001187"

UBERON_IDS_Stomach-AdenoCA := "UBERON:0000945"
REPLISEQ_Stomach-AdenoCA := "EFO:0001187"

UBERON_IDS_Skin-Melanoma := "CL:1000458" "UBERON:0002097" "UBERON:0001003" "UBERON:0004264" "UBERON:0036149"
REPLISEQ_Skin-Melanoma := "EFO:0001196"

UBERON_IDS_Liver-HCC := "UBERON:0002107" "UBERON:0001115" "UBERON:0001114"
REPLISEQ_Liver-HCC := "EFO:0001187"

UBERON_IDS_ColoRect-AdenoCA := "UBERON:0000317" "UBERON:0004992" "UBERON:0001159" "UBERON:0001157" "UBERON:0008971"
REPLISEQ_ColoRect-AdenoCA := "EFO:0001187"

UBERON_IDS_Lymph-All := "CL:0000084" "NTR:0000537"
REPLISEQ_Lymph-All := "EFO:0002784"

UBERON_IDS_Head-SCC := "UBERON:0006920"
REPLISEQ_Head-SCC := "EFO:0001196"

UBERON_IDS_Eso-AdenoCA := "UBERON:0001043" "UBERON:0002469" "UBERON:0004648"
REPLISEQ_Eso-AdenoCA := "EFO:0001187"

UBERON_IDS_CNS-GbmOligoPiloAstro := "CL:0000127" "CL:0002603" "CL:0002604" "CL:0002606" "EFO:0003072"
REPLISEQ_CNS-GbmOligoPiloAstro := "EFO:0003072"

UBERON_IDS_Uterus-AdenoCA := "UBERON:0000995" "EFO:0002791"
REPLISEQ_Uterus-AdenoCA := "EFO:0002791"

UBERON_IDS_Cervix-All := "UBERON:0000995" "EFO:0002791"
REPLISEQ_Cervix-All := "EFO:0002791"

UBERON_IDS_Bone-Osteosarc := "CL:0000062" "CL:0002540" "CL:0010001" "UBERON:0002371" "CL:0001059"
REPLISEQ_Bone-Osteosarc := "EFO:0002784"

UBERON_IDS_Bladder-AdenoCA := "UBERON:0001255" "UBERON:0001259" "UBERON:0005033" "CL:2000040"
REPLISEQ_Bladder-AdenoCA := "EFO:0001187"


configs/repliseq.yaml: download_encode.py
	python download_encode.py --genome hg19 --assays "Repli-seq" --no-groupby > $@

configs/functional_data.yaml: download_encode.py
	python download_encode.py > $@

configs/experiments.yaml: configs/repliseq.yaml configs/functional_data.yaml
	cat $^ > $@

uberon_terms/%: configs/experiments.yaml
	python get_uberon_terms.py $< $*

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

# Use grouped targets (&:) to tell Make these files are created together atomically
gtensors/%.features: configs/features/%.yaml
	@mkdir -p $(dir $@)
	gtensor compose base_config.yaml configs/features/$*.yaml -o gtensors/$* -w 3 && \
	gtensor feature ls gtensors/$*.nc > gtensors/$*.features 

gtensors/%.samples: gtensors/%.features
	@mkdir -p $(dir $@)
	gtensor compose base_config.yaml configs/features/$*.yaml configs/samples/$*.yaml -o gtensors/$* -w 10 && \
	gtensor info gtensors/$*.nc > $@

staged/%.train.nc staged/%.test.nc: gtensors/%.samples
	@mkdir -p staged
	gtensor split -o staged/$*. --min-mutations 800 --min-region-size 100 gtensors/$*.nc "chr2"

# Optional arguments that can be passed via command line
STUDY_ARGS_Lung-All := -min 10 -max 20 -lsub 0.125 -init SBS17a -init SBS17b # 02
STUDY_ARGS_Ovary-AdenoCA := -min 5 -max 15 -lsub 0.125 -init SBS1 -init SBS2 -init SBS5 -init SBS8 -init SBS13 -init SBS18 # 02
STUDY_ARGS_Jake-Breast := -min 8 -max 23 -lsub 0.125 -bsub 0.5 -init SBS1 -init SBS2 -init SBS5 -init SBS8 -init SBS13 -init SBS17a -init SBS17b -init SBS18 # 01

studies/%/: staged/$$(firstword $$(subst /, ,$$*)).train.nc staged/$$(firstword $$(subst /, ,$$*)).test.nc
	@mkdir -p studies
	$(if $(STUDY_ARGS_$(firstword $(subst /, ,$*))),, \
		$(error Error: STUDY_ARGS_$(firstword $(subst /, ,$*)) is not defined. Please define it in the Makefile))
	topo-model study create $@ -ds $^ $(STUDY_ARGS_$(firstword $(subst /, ,$*))) --save-model -ee -stop 100

stage.all: staged/Lung-All.train.nc staged/Ovary-AdenoCA.train.nc staged/Jake-Breast.train.nc \
	staged/Prost-AdenoCA.train.nc staged/Myeloid-All.train.nc staged/Panc-AdenoCA.train.nc \
	staged/Kidney-All.train.nc staged/Panc-Endocrine.train.nc staged/Stomach-AdenoCA.train.nc \
	staged/Skin-Melanoma.train.nc staged/Liver-HCC.train.nc staged/ColoRect-AdenoCA.train.nc \
	staged/Lymph-All.train.nc staged/Head-SCC.train.nc staged/Eso-AdenoCA.train.nc \
	staged/CNS-GbmOligoPiloAstro.train.nc staged/Uterus-AdenoCA.train.nc \
	staged/Cervix-All.train.nc staged/Bone-Osteosarc.train.nc staged/Bladder-AdenoCA.train.nc

run/%/: studies/%/
	topo-model study run $< -@ 5 --lazy

array/%/: studies/%/
	sbatch \
		--mem-per-cpu=1100 \
		--cpus-per-task=5 \
		--ntasks=1 \
		--time=6:00:00 \
		--job-name=$< \
		--array=1-5%25 \
		--output=logs/%x/%A_%a.out \
		--partition=short,park \
		--account=park \
		--wrap="topo-model study run $< -@ 4 --lazy --time-limit 710"

analyses/%/summary.png: studies/%/
	@mkdir -p $(dir $@)
	python bin/plot_summary.py $< $@ && \
	code -r $@

analyses/%/analysis.ipynb: bin/analysis_template.ipynb
	@mkdir -p $(dir $@)
	@# Parse the target path: analyses/Ovary-AdenoCA/02/1/analysis.ipynb
	@# Extract: study_name=Ovary-AdenoCA, study_id=02, model_id=1
	@FULL_PATH="$*"; \
	STUDY_NAME=$$(echo $$FULL_PATH | cut -d'/' -f1); \
	STUDY_ID=$$(echo $$FULL_PATH | cut -d'/' -f2); \
	MODEL_ID=$$(echo $$FULL_PATH | cut -d'/' -f3); \
	STUDY_DIR="studies/$$STUDY_NAME/$$STUDY_ID"; \
	job_id="analyses/$${STUDY_NAME}/$${STUDY_ID}/$${MODEL_ID}"; \
	sbatch \
		--mem=10G \
		--cpus-per-task=5 \
		--ntasks=1 \
		--time=1:00:00 \
		--job-name=$$job_id \
		--output=%x/log.out \
		--partition=short,park \
		--account=park \
		--wait \
		--wrap="papermill bin/analysis_template.ipynb $@ \
			-p study_name $$STUDY_NAME \
			-p study_id $$STUDY_ID \
			-p model_id $$MODEL_ID \
			-p study_dir $$STUDY_DIR"