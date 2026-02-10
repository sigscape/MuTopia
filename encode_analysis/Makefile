
# Enable secondary expansion for multi-wildcard rules
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -e -u -c

.SECONDEXPANSION:
# Prevent Make from deleting config files when downstream targets fail
.PRECIOUS: configs/features/%.yaml staged/%.train.nc staged/%.test.nc studies/%/ gtensors/%.nc gtensors/%.features gtensors/%.samples analyses/%/annotated.nc configs/samples/%.yaml

# Phony targets for convenience - allows "make lung" instead of "make data/gtensors/lung.done"
.PHONY: run/% uberon_terms/% summarize/%/ stage.all retrain/% pickup_retrain/% rm_blacklist setup_studies run_analyses/%/

# Define tissue-specific UBERON IDs and repliseq terms
UBERON_IDS_Lung-All := "UBERON:0002048" "UBERON:0002168" "UBERON:0002167" "UBERON:0008953" "UBERON:0008952" "UBERON:0002170" "UBERON:0002171"
REPLISEQ_Lung-All := "EFO:0001196"

UBERON_IDS_Ovary-AdenoCA := "UBERON:0000992"
REPLISEQ_Ovary-AdenoCA := "EFO:0001203"

UBERON_IDS_Jake-Breast := "UBERON:0008367" "UBERON:0000310" "CL:0002327"seq
REPLISEQ_Jake-Breast := "EFO:0001203"

UBERON_IDS_Breast-All := "UBERON:0008367" "UBERON:0000310" "CL:0002327"
REPLISEQ_Breast-All := "EFO:0001203"

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

UBERON_IDS_Skin-Melanoma := "CL:1000458" "UBERON:0002097" "UBERON:0001003" "UBERON:0004264" "UBERON:0036149" "CL:1001606"
REPLISEQ_Skin-Melanoma := "EFO:0001196"

UBERON_IDS_Liver-HCC := "UBERON:0002107" "UBERON:0001115" "UBERON:0001114"
REPLISEQ_Liver-HCC := "EFO:0001187"

UBERON_IDS_ColoRect-AdenoCA := "UBERON:0000317" "UBERON:0004992" "UBERON:0001159" "UBERON:0001157" "UBERON:0008971"
REPLISEQ_ColoRect-AdenoCA := "EFO:0001187"

UBERON_IDS_Lymph-All := "CL:0000084" "NTR:0000537"
REPLISEQ_Lymph-All := "EFO:0002784"

UBERON_IDS_Lymph-CLL := "CL:0000084" "NTR:0000537"
REPLISEQ_Lymph-CLL := "EFO:0002784"

UBERON_IDS_Lymph-BNHL := "CL:0000084" "NTR:0000537"
REPLISEQ_Lymph-BNHL := "EFO:0002784"

UBERON_IDS_Head-SCC := "UBERON:0006920"
REPLISEQ_Head-SCC := "EFO:0001196"

UBERON_IDS_Eso-AdenoCA := "UBERON:0001043" "UBERON:0002469" "UBERON:0004648"
REPLISEQ_Eso-AdenoCA := "EFO:0001187"

UBERON_IDS_CNS-All := "CL:0000127" "CL:0002603" "CL:0002604" "CL:0002606" "EFO:0003072"
REPLISEQ_CNS-All := "EFO:0003072"

UBERON_IDS_CNS-GBM := "CL:0000127" "CL:0002603" "CL:0002604" "CL:0002606" "EFO:0003072"
REPLISEQ_CNS-GBM := "EFO:0003072"

UBERON_IDS_CNS-Medullo := "CL:0011020" "CL:0000100"
REPLISEQ_CNS-Medullo := "EFO:0003072"

UBERON_IDS_Uterus-AdenoCA := "UBERON:0000995" "EFO:0002791"
REPLISEQ_Uterus-AdenoCA := "EFO:0002791"

UBERON_IDS_Cervix-All := "UBERON:0000995" "EFO:0002791"
REPLISEQ_Cervix-All := "EFO:0002791"

UBERON_IDS_Bone-Osteosarc := "CL:0000062" "CL:0002540" "CL:0010001" "UBERON:0002371" "CL:0001059"
REPLISEQ_Bone-Osteosarc := "EFO:0002784"

UBERON_IDS_Bladder-TCC := "UBERON:0001255" "UBERON:0001259" "UBERON:0005033" "CL:2000040"
REPLISEQ_Bladder-TCC := "EFO:0001187"

configs/repliseq.yaml:
	python bin/download_encode.py --genome hg19 --assays "Repli-seq" --no-groupby > $@

configs/functional_data.yaml:
	python bin/download_encode.py > $@

configs/experiments.yaml: configs/repliseq.yaml configs/functional_data.yaml
	cat $^ > $@

uberon_terms/%: configs/experiments.yaml
	python bin/get_uberon_terms.py $< $*

# Pattern rule that uses tissue-specific variables
configs/features/%.yaml: configs/experiments.yaml base_config.yaml
	@mkdir -p $(dir $@)
	python bin/write_config.py \
		base_config.yaml \
		$< \
		--name $* \
		-ids $(UBERON_IDS_$*) \
		-repliseq $(REPLISEQ_$*) \
		-o $@

configs/samples/%.yaml:
	@mkdir -p $(dir $@)
	python bin/write_sample_config.py $*

# Use grouped targets (&:) to tell Make these files are created together atomically
gtensors/%.features: configs/features/%.yaml
	@mkdir -p $(dir $@)
	gtensor compose base_config.yaml configs/features/$*.yaml -o gtensors/$* -w 4 && \
	gtensor feature ls gtensors/$*.nc > gtensors/$*.features 

gtensors/%.samples: gtensors/%.features configs/samples/%.yaml
	@mkdir -p $(dir $@)
	gtensor compose base_config.yaml configs/features/$*.yaml configs/samples/$*.yaml -o gtensors/$* -w 10 && \
	gtensor info gtensors/$*.nc > $@

staged/%.train.nc staged/%.test.nc: gtensors/%.samples
	mkdir -p staged && \
	gtensor split -o staged/$*. --min-mutations 400 --min-region-size 100 gtensors/$*.nc "chr2" \
		&& gtensor feature rm staged/$*.train.nc "AccessiblePeak"

studies/%/presets.01/: staged/%.train.nc staged/%.test.nc
	@mkdir -p studies/$*/presets.01
	PARAMS=$$(python bin/set_parameters.py $* 25); \
	topo-model study create $@ -ds $^ -lsub 0.2 -creg 0.00025 --save-model -ee -stop 100 $$PARAMS

studies/%/presets.02/: staged/%.train.nc staged/%.test.nc
	@mkdir -p studies/$*/presets.02
	PARAMS=$$(python bin/params_from_config.py $*); \
	topo-model study create $@ -ds $^ -lsub 0.2 -creg 0.00025 --save-model -ee -stop 100 $$PARAMS

studies/%/presets.04/: staged/%.train.nc staged/%.test.nc
	@mkdir -p studies/$*/presets.02
	PARAMS=$$(python bin/params_from_config.py $*); \
	topo-model study create $@ -ds $^ -lsub 0.2 --save-model -ee -stop 50 $$PARAMS

setup_studies/%:
	for tt in `cat tumor_types.txt`; do \
		$(MAKE) studies/$$tt/$*; \
	done

run/%/: studies/%/
	topo-model study run $< -@ 1 --lazy

BLACKLIST := "68c2a355-862c-4657-b296-5776ed8447b0" "7dc5f8ba-0080-43d3-8426-bd527a970761" "81b1e78c-6032-4ff4-b52a-83456b9450ea" "4a4e0397-8702-46ef-aaa8-3980f03a40bf" "b2ec0fd0-fbcf-4abc-ad80-4ae444e30b55" "e41bc2ec-3e0b-4c37-806b-3f6f25c8c4db" "f98de26b-c7d6-435d-81fa-1f1869da9087" "faff4626-615b-416a-b7a6-9d177dcc94a9" "5b180356-cf58-4fad-a3d4-00fc12b43fcc" "03cff38d-7e29-4409-a508-749bddb1b3df" "bc1d5327-2e76-4e0e-b749-72a559469d0d"

rm_blacklist:
	for tt in `cat tumor_types.txt`; do \
		gtensor sample rm staged/$$tt.train.nc $(BLACKLIST); \
		gtensor sample rm staged/$$tt.test.nc $(BLACKLIST); \
	done

testrun/%/: studies/%/
	@FULL_PATH="$*"; \
	STUDY_NAME=$$(echo $$FULL_PATH | cut -d'/' -f1); \
	sbatch \
		--mem-per-cpu=3500 \
		--cpus-per-task=1 \
		--ntasks=1 \
		--time=10:00 \
		--job-name=$< \
		--array=1-3 \
		--output=logs/%x/%A_%a.out \
		--partition=short,park \
		--account=park \
		--wrap="topo-model study run $< -@ 1 --lazy --time-limit 8"

array/%/: studies/%/
	@FULL_PATH="$*"; \
	STUDY_NAME=$$(echo $$FULL_PATH | cut -d'/' -f1); \
	sbatch \
		--mem-per-cpu=7500 \
		--cpus-per-task=1 \
		--ntasks=1 \
		--time=6:00:00 \
		--job-name=$< \
		--array=1-100%5 \
		--output=logs/%x/%A_%a.out \
		--partition=short,park \
		--account=park \
		--wrap="topo-model study run $< -@ 1 --lazy --time-limit 350"

retrain/%:
	@FULL_PATH="$*"; \
	STUDY_NAME=$$(echo $$FULL_PATH | cut -d'/' -f1); \
	STUDY_ID=$$(echo $$FULL_PATH | cut -d'/' -f2); \
	MODEL_ID=$$(echo $$FULL_PATH | cut -d'/' -f3); \
	sbatch \
		--mem-per-cpu=5000 \
		--cpus-per-task=1 \
		--ntasks=1 \
		--time=6:00:00 \
		--job-name=$@ \
		--output=logs/%x/%A_%a.out \
		--partition=short,park \
		--account=park \
		--wrap="topo-model study retrain "studies/$$STUDY_NAME/$$STUDY_ID" $$MODEL_ID "studies/$$STUDY_NAME/$$STUDY_ID/trial=$$MODEL_ID.pkl" -@ 1 --lazy --time-limit 350"

pickup_retrain/%:
	FULL_PATH="$*"; \
	STUDY_NAME=$$(echo $$FULL_PATH | cut -d'/' -f1); \
	STUDY_ID=$$(echo $$FULL_PATH | cut -d'/' -f2); \
	PICKUP_MODELS=$$(python bin/get_models_to_pickup.py studies/$$STUDY_NAME/$$STUDY_ID); \
	if [ -z "$$PICKUP_MODELS" ]; then \
		echo "No models to pickup for $$STUDY_NAME/$$STUDY_ID"; \
		exit 0; \
	fi; \
	ARRAY_SPEC=$$(echo $$PICKUP_MODELS | tr ' ' ','); \
	sbatch \
		--mem-per-cpu=7500 \
		--cpus-per-task=1 \
		--ntasks=1 \
		--time=6:00:00 \
		--job-name=pickup_retrain_$$STUDY_NAME/$$STUDY_ID \
		--array=$$ARRAY_SPEC%10 \
		--output=logs/pickup_retrain/$$STUDY_NAME/$$STUDY_ID/%A_%a.out \
		--partition=short,park \
		--account=park \
		--wrap="topo-model study retrain studies/$$STUDY_NAME/$$STUDY_ID \$$SLURM_ARRAY_TASK_ID studies/$$STUDY_NAME/$$STUDY_ID/trial=\$$SLURM_ARRAY_TASK_ID.pkl -@ 1 --lazy --time-limit 350"

analyses/%/summary.png:
	@mkdir -p $(dir $@)
	python bin/plot_summary.py studies/$* $@ && \
	code -r $@

analyses/%/spectra.pdf:
	@mkdir -p $(dir $@)
	python bin/plot_spectra.py studies/$* $@ && \
	code -r $@

analyses/%/annotated.nc:
	@mkdir -p $(dir $@)
	@FULL_PATH="$*"; \
	STUDY_NAME=$$(echo $$FULL_PATH | cut -d'/' -f1); \
	STUDY_ID=$$(echo $$FULL_PATH | cut -d'/' -f2); \
	MODEL_ID=$$(echo $$FULL_PATH | cut -d'/' -f3); \
	STUDY_DIR="studies/$$STUDY_NAME/$$STUDY_ID"; \
	sbatch \
		--mem=10G \
		--cpus-per-task=5 \
		--ntasks=1 \
		--time=1:00:00 \
		--job-name=$@ \
		--output=%x.log \
		--partition=short,park \
		--account=park \
		--wait \
		--wrap="topo-model annot \
			studies/$$STUDY_NAME/$$STUDY_ID/trial=$$MODEL_ID.pkl \
			gtensors/$$STUDY_NAME.nc \
			$@ \
			--region chr2 \
			-@ 5"

analyses/%/annotated.shap.nc:
	@mkdir -p $(dir $@)
	@FULL_PATH="$*"; \
	STUDY_NAME=$$(echo $$FULL_PATH | cut -d'/' -f1); \
	STUDY_ID=$$(echo $$FULL_PATH | cut -d'/' -f2); \
	MODEL_ID=$$(echo $$FULL_PATH | cut -d'/' -f3); \
	STUDY_DIR="studies/$$STUDY_NAME/$$STUDY_ID"; \
	sbatch \
		--mem=10G \
		--cpus-per-task=10 \
		--ntasks=1 \
		--time=1:00:00 \
		--job-name=$@ \
		--output=%x.log \
		--partition=short,park \
		--account=park \
		--wait \
		--wrap="topo-model annot \
			studies/$$STUDY_NAME/$$STUDY_ID/trial=$$MODEL_ID.pkl \
			gtensors/$$STUDY_NAME.nc \
			--calc-shap \
			$@ \
			--region chr2 \
			-@ 10"

analyses/%/analysis.ipynb: analyses/%/annotated.nc
	@mkdir -p $(dir $@)
	sbatch --mem=5G \
		--cpus-per-task=2 \
		--ntasks=1 \
		--time=30:00 \
		--job-name=$@ \
		--output=%x.log \
		--partition=short,park \
		--account=park \
		--wait \
		--wrap="papermill bin/analysis_template.ipynb $@ -p data_path $<"

run_analyses/%/:
	for tt in `cat tumor_types.txt`; do \
		models=$$(python bin/get_analysis_models.py studies/$$tt/$*/); \
		for model in $$models; do \
			$(MAKE) analyses/$$tt/$*/$$model/analysis.ipynb & \
		done; \
	done

# use studies.txt to get all the studies and tar their "summary.png" and "spectra.pdf"
training_summary.tar.gz:
	tar -czvf $@ $$(for tt in $$(cat tumor_types.txt); do find analyses/$$tt/presets.04/ -type f ! -name "*.nc" ! -name "*.log" ! -name "*.csv.gz"; done)

analyses/%/base_annotation.nc:
	@mkdir -p $(dir $@)
	@FULL_PATH="$*"; \
	STUDY_NAME=$$(echo $$FULL_PATH | cut -d'/' -f1); \
	STUDY_ID=$$(echo $$FULL_PATH | cut -d'/' -f2); \
	MODEL_ID=$$(echo $$FULL_PATH | cut -d'/' -f3); \
	STUDY_DIR="studies/$$STUDY_NAME/$$STUDY_ID"; \
	sbatch \
		--mem=5G \
		--cpus-per-task=5 \
		--ntasks=1 \
		--time=30:00 \
		--job-name=$@ \
		--output=%x.log \
		--partition=short,park \
		--account=park \
		--wait \
		--wrap="python bin/base_annotation.py --model studies/$$STUDY_NAME/$$STUDY_ID/trial=$$MODEL_ID.pkl --base-gtensor gtensors/Lung-All.nc --output $@"

joint_summary.tar.gz:
	tar -cvf $@ $$(for model in $$(cat models.txt); do echo "analyses/$$model/annotated.shap.nc" "analyses/$$model/base_annotation.nc analyses/$$model/analysis.ipynb"; done) meta_analysis.ipynb models.txt

joint_summary.doga.tar.gz:
	tar -cvf $@ $$(for model in $$(cat models.txt); do echo "analyses/$$model/base_annotation.nc*.gz analyses/$$model/annotated.shap.nc"; done) analyses/joint_collection/*

