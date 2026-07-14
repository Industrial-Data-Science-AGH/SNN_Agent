# T01 — provider + remote state backend + resource group.
#
# See docs/plans/T01-azure-infra-and-cd-pipeline.md and
# docs/architecture/features/F06-deployment-pipeline/design.md for the
# one-time manual bootstrap this depends on: the Storage Account + `tfstate`
# container backing the `backend "azurerm"` block below must exist BEFORE
# `terraform init` can use them (chicken-and-egg — bootstrapped by hand or
# via one local `terraform apply` with the backend block commented out,
# per the runbook).

terraform {
  required_version = ">= 1.5"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
  }

  # Values intentionally left out (partial config) — the state Storage
  # Account name is created by hand during bootstrap and varies per owner,
  # so it must never be hardcoded into a committed file (ADR-0010, ADR-0011).
  # Supplied at `terraform init` time via `-backend-config` flags; see
  # .github/workflows/deploy.yml.
  backend "azurerm" {
    container_name = "tfstate"
    key             = "dashboard.tfstate"
  }
}

provider "azurerm" {
  features {}
}

locals {
  # Single tag set applied to every resource this module creates, so
  # `dev`/`prod` throwaway vs. real resources are easy to tell apart in the
  # Azure portal and in `az resource list`.
  tags = {
    project     = "wakeup-ai-cloud-dashboard"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# Not called out as its own file in the T01 plan's "Files to create" list,
# but every resource below needs a resource group to live in — kept here
# rather than inventing a fifth .tf file for one resource.
resource "azurerm_resource_group" "main" {
  name     = "rg-snn-agents-${var.environment}"
  location = var.location
  tags     = local.tags
}
