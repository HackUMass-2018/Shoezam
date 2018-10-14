.PHONY: training-ssh

# training-ssh opens an ssh connection to the GCP training instance
training-ssh:
	gcloud compute --project "shoezam-2018" ssh --zone "us-east1-b" "ubuntu-training"
