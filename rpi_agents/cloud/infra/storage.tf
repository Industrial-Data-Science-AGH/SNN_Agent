# T01 — Storage Account: Blob container `snapshots` (F02 images), Table
# `events` (F02 metadata), and the private `tfstate` container backing this
# module's own remote state (ADR-0010, ADR-0011).
#
# Deliberately one Storage Account for all three, not separate resources —
# see F02 design (Context) and ADR-0007's "Storage choice" section: minimizes
# both cost and the number of things to provision/secure at this project's
# volume.

resource "azurerm_storage_account" "main" {
  # Storage Account names must be globally unique, 3-24 chars, lowercase
  # letters/numbers only. If this is already taken, override with a
  # `-var` in a follow-up rather than editing this file per-owner.
  name                = "snnagents${var.environment}"
  resource_group_name = azurerm_resource_group.main.name
  location            = azurerm_resource_group.main.location

  account_tier             = "Standard"
  account_replication_type = "LRS"
  min_tls_version           = "TLS1_2"

  # Blob container "snapshots" (F02) must stay private — no public/anonymous
  # read access at the container level; reads only ever happen through
  # short-lived SAS URLs minted by the app (F01 design).
  allow_nested_items_to_be_public = false

  tags = local.tags
}

resource "azurerm_storage_container" "snapshots" {
  name                  = "snapshots"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}

resource "azurerm_storage_table" "events" {
  name                 = "events"
  storage_account_name = azurerm_storage_account.main.name
}

# Backs this module's own `backend "azurerm"` block (main.tf). Bootstrapped
# once, by hand, before the first `terraform init` (chicken-and-egg — this
# resource can't provision the backend it needs to run). Left here, managed
# by Terraform from that point on, so its lifecycle isn't split across a
# manual step and a Terraform resource forever.
resource "azurerm_storage_container" "tfstate" {
  name                  = "tfstate"
  storage_account_name  = azurerm_storage_account.main.name
  container_access_type = "private"
}
