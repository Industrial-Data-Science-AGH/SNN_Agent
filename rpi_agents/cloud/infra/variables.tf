# T01 — input variables.
#
# Sensitive variables (dashboard_password, storage_connection_string,
# ghcr_pull_token) are marked `sensitive = true` so Terraform never prints
# their values in plan/apply logs (ADR-0010). They are passed in from
# GitHub Actions repository secrets as `-var` flags — never given a default
# here, never committed as a literal value anywhere in this module.

variable "environment" {
  description = "Deployment environment. Parameterizes resource names so a throwaway dev environment can exist alongside prod without duplicating this module (F02 design)."
  type        = string

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be \"dev\" or \"prod\"."
  }
}

variable "location" {
  description = "Azure region for all resources."
  type        = string
  default     = "germanywestcentral"
}

variable "compute_location" {
  description = "Azure region for the Container Apps environment + Log Analytics workspace. Separate from `location` because this subscription's Container Apps environment quota is exhausted in some regions (e.g. germanywestcentral) independently of where the resource group / storage account live."
  type        = string
  default     = "francecentral"
}

variable "image_tag" {
  description = "Container image tag to deploy — the git SHA the GitHub Actions workflow just built and pushed to ghcr.io (ADR-0012)."
  type        = string
}

variable "github_repository" {
  description = "GitHub \"owner/repo\" slug (GitHub Actions' repository context value), used to build the ghcr.io image reference and the registry pull username."
  type        = string
}

variable "dashboard_user" {
  description = "Non-secret HTTP Basic Auth username (ADR-0009). Defaults to \"ids\" per the accepted default; change it before exposing the app publicly."
  type        = string
  default     = "ids"
}

variable "dashboard_password" {
  description = "HTTP Basic Auth password shared by the dashboard and the Pi push client (ADR-0009). Sourced from the GitHub Actions repo secret DASHBOARD_PASSWORD."
  type        = string
  sensitive   = true
}

variable "storage_connection_string" {
  description = "Connection string for the Storage Account (azurerm_storage_account.main), injected into the Container App as a native secret so neither the dashboard page nor the Pi ever holds it directly (F02 design, Security)."
  type        = string
  sensitive   = true
}

variable "ghcr_pull_token" {
  description = "GitHub PAT, scoped read:packages only, used by the Container App to pull the image from ghcr.io (ADR-0012). Sourced from the GitHub Actions repo secret GHCR_PULL_TOKEN."
  type        = string
  sensitive   = true
}
