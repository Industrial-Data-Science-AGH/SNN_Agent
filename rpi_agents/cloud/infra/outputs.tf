# T01 — non-sensitive outputs only (F02/F06 design). Never output the
# storage connection string or the dashboard password — both are sensitive
# variables and stay inside Terraform state, never printed in plan/apply
# logs or surfaced here.

output "container_app_fqdn" {
  description = "Public FQDN of the deployed Container App (dashboard + ingest API)."
  value       = azurerm_container_app.dashboard.latest_revision_fqdn
}

output "storage_account_name" {
  description = "Name of the Storage Account holding the `snapshots` container and `events` table."
  value       = azurerm_storage_account.main.name
}
