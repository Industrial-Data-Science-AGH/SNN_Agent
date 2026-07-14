# T01 — Container App Environment + Container App: the runtime for F01
# (ingest API), F04 (dashboard UI), F05 (Basic Auth) — Consumption plan,
# scale-to-zero (ADR-0007). T01 itself only needs the placeholder image in
# cloud/app/ to deploy successfully; T02 replaces the app code behind this
# same shape.

# azurerm_container_app_environment requires a Log Analytics workspace to
# ship its (minimal, free-tier-eligible) platform logs to.
resource "azurerm_log_analytics_workspace" "main" {
  name                = "log-snn-agents-${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = local.tags
}

resource "azurerm_container_app_environment" "main" {
  name                       = "cae-snn-agents-${var.environment}"
  resource_group_name       = azurerm_resource_group.main.name
  location                   = azurerm_resource_group.main.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id

  tags = local.tags
}

resource "azurerm_container_app" "dashboard" {
  name                         = "ca-snn-agents-${var.environment}"
  resource_group_name         = azurerm_resource_group.main.name
  container_app_environment_id = azurerm_container_app_environment.main.id
  revision_mode                 = "Single"

  # PAT-based pull credential (ADR-0012) — the push side uses the
  # automatic GITHUB_TOKEN instead (see .github/workflows/deploy.yml), so
  # this is the only registry credential Terraform needs to manage.
  registry {
    server               = "ghcr.io"
    username              = split("/", var.github_repository)[0]
    password_secret_name = "ghcr-pull-token"
  }

  secret {
    name  = "ghcr-pull-token"
    value = var.ghcr_pull_token
  }

  # Native Container App secret storage (encrypted at rest), referenced by
  # `env` blocks below — distinct from the plain DASHBOARD_USER env var
  # (ADR-0010).
  secret {
    name  = "dashboard-password"
    value = var.dashboard_password
  }

  secret {
    name  = "storage-connection-string"
    value = var.storage_connection_string
  }

  template {
    min_replicas = 0
    max_replicas = 1

    container {
      name   = "dashboard"
      image  = "ghcr.io/${var.github_repository}/dashboard:${var.image_tag}"
      cpu    = 0.25
      memory = "0.5Gi"

      env {
        name  = "DASHBOARD_USER"
        value = var.dashboard_user
      }

      env {
        name        = "DASHBOARD_PASSWORD"
        secret_name = "dashboard-password"
      }

      env {
        name        = "AZURE_STORAGE_CONNECTION_STRING"
        secret_name = "storage-connection-string"
      }
    }
  }

  ingress {
    external_enabled = true
    target_port       = 8000
    transport          = "auto"

    traffic_weight {
      latest_revision = true
      percentage       = 100
    }
  }

  tags = local.tags
}
