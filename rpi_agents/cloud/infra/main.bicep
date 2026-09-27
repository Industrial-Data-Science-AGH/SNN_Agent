// SNN Agent backend on Azure. One resource group holds everything, so `az group delete` removes all of it.
//
// What it creates: Log Analytics, a user-assigned managed identity, a container registry, a storage account (tables,
// blobs, queues), a Key Vault, an Azure AI Services account with a vision deployment (gpt-5-mini by default), a
// Container Apps environment and two apps (API + worker) that run the same image.
//
// Rollout is two phases because the apps need an image in the registry and secrets in the vault before they can start:
//   1. deployApps=false  -> everything except the apps
//   2. push the image, put the secrets in the vault (see deploy.sh)
//   3. deployApps=true   -> the apps
//
// Choices worth knowing about:
// - No keys anywhere. Storage has shared-key access off, the AI account has local (key) auth off, the registry has the
//   admin user off. The workloads use one managed identity with the narrow roles below.
// - Secrets live in Key Vault only. The apps hold `keyvault:<name>` markers and read the values themselves at start-up with
//   the managed identity (keyvault.py), so no secret is in this template, the parameters, the deployment history or the app
//   definition. (Container Apps' own Key Vault references are refused by "express" environments, which a students
//   subscription gets.) Rotating a secret = new value in the vault + a restart.
// - A future dashboard or the real SNN runtime is another container app in the same environment. It reuses the registry,
//   the vault, the storage account and the environment; give it its own identity and the read-only roles it needs.

targetScope = 'resourceGroup'

@description('Prefix for resource names, 3-12 lowercase letters or digits.')
@minLength(3)
@maxLength(12)
param prefix string = 'snnbe'

param location string = resourceGroup().location

@description('Environment label used in tags and names of things that must be told apart (dev, prod).')
param environmentName string = 'dev'

param tags object = { project: 'snn-agent', component: 'backend', environment: environmentName }

@description('Create the container apps. False for the first phase, before the image and the secrets exist.')
param deployApps bool = true

@description('Repository and tag inside the registry, for example snn-backend:1.0.0.')
param imageName string = 'snn-backend:latest'

@description('Object id of the person who operates this (roles for the vault and the data). Empty = none.')
param adminPrincipalId string = ''

@description('Extra public host names of the API, comma separated (for a custom domain). The default host is always allowed.')
param extraAllowedHosts string = ''

param operatorUsername string = 'operator'

@description('Runtime factory as package.module:factory, or "demo" (a stand-in that is not an SNN).')
param runtime string = 'demo'
param allowDemoRuntime bool = true

@description('Path of the model manifest inside the image.')
param manifestPath string = '/app/contracts/fixtures/model-manifest.json'

@allowed(['manual-review-only-v1', 'armed-glass-and-person-v1'])
param policy string = 'manual-review-only-v1'

param allowLive bool = false

@description('Create an AI Services account and a vision deployment. Set false to point at an existing endpoint instead.')
param deployModel bool = true
param modelName string = 'gpt-5-mini'
param modelVersion string = '2025-08-07'
@description('Thousands of tokens per minute. Students subscriptions have small quotas; this is far more than needed.')
param modelCapacity int = 10
@allowed(['chat', 'reasoning'])
param visionFamily string = 'reasoning'
@description('Only when deployModel is false.')
param externalVisionEndpoint string = ''
param externalVisionDeployment string = ''

@description('Send e-mail. Needs the secrets smtp-user, smtp-password, smtp-from and alert-recipients in the vault.')
param mailEnabled bool = true
param smtpHost string = 'smtp.gmail.com'
param smtpPort int = 587

@description('Resource id of an existing Container Apps environment to join. Empty = create one. A subscription may be limited to a single environment (Azure for Students is), and joining an existing one also puts the apps next to a dashboard that already lives there.')
param existingEnvironmentId string = ''

param apiMinReplicas int = 1
param apiMaxReplicas int = 2
param workerMaxReplicas int = 2

@description('Scale the worker on the vision queue length (KEDA, with the identity). Express environments refuse custom scale rules, so it is off by default: one always-on worker.')
param scaleWorkerOnQueue bool = false

var suffix = uniqueString(resourceGroup().id, prefix)
var storageName = take('${prefix}${suffix}', 24)
var vaultName = take('${prefix}-kv-${suffix}', 24)
var registryName = take('${prefix}acr${suffix}', 50)
var aiName = take('${prefix}-ai-${suffix}', 60)
var apiName = '${prefix}-api'
var workerName = '${prefix}-worker'

var roles = {
  tableContributor: '0a9a7e1f-b9d0-4cc4-a60d-0319b160aaa3'
  blobContributor: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  queueContributor: '974c5e8b-45b9-4653-ba55-5f855dd0fb88'
  vaultSecretsUser: '4633458b-17de-408a-b874-0445c86b69e6'
  vaultSecretsOfficer: 'b86a8fe4-44ce-4948-aee5-eccb2c155cd7'
  acrPull: '7f951dda-4ed3-4680-a7ca-43fe172d538d'
  openAiUser: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
}

var createEnvironment = empty(existingEnvironmentId)

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = if (createEnvironment) {
  name: '${prefix}-logs-${suffix}'
  location: location
  tags: tags
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    workspaceCapping: { dailyQuotaGb: json('0.5') } // a runaway log cannot eat a student credit
  }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: '${prefix}-id-${environmentName}'
  location: location
  tags: tags
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: registryName
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: false }
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  tags: tags
  kind: 'StorageV2'
  sku: { name: 'Standard_LRS' }
  properties: {
    allowSharedKeyAccess: false
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    defaultToOAuthAuthentication: true
  }
}

// Created here so the identity does not need rights to create them; the adapter also creates any that are missing.
resource tableService 'Microsoft.Storage/storageAccounts/tableServices@2023-05-01' = { parent: storage, name: 'default' }
resource tables 'Microsoft.Storage/storageAccounts/tableServices/tables@2023-05-01' = [for name in [
  'devices', 'sessions', 'batches', 'events', 'devicecommands', 'requests', 'visionruns', 'devicestatus', 'websessions'
]: {
  parent: tableService
  name: name
}]

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = { parent: storage, name: 'default' }
resource images 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'images'
  properties: { publicAccess: 'None' }
}

resource queueService 'Microsoft.Storage/storageAccounts/queueServices@2023-05-01' = { parent: storage, name: 'default' }
resource queues 'Microsoft.Storage/storageAccounts/queueServices/queues@2023-05-01' = [for name in [
  'vision-jobs', 'vision-jobs-poison', 'notify-jobs', 'notify-jobs-poison'
]: {
  parent: queueService
  name: name
}]

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: vaultName
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7 // the shortest; a students subscription should not pin a name for 90 days
  }
}

resource ai 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (deployModel) {
  name: aiName
  location: location
  tags: tags
  kind: 'AIServices'
  sku: { name: 'S0' }
  properties: {
    customSubDomainName: aiName
    disableLocalAuth: true // identity only, no API keys to leak or rotate
    publicNetworkAccess: 'Enabled'
  }
}

resource vision 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (deployModel) {
  parent: ai
  name: modelName
  sku: { name: 'GlobalStandard', capacity: modelCapacity }
  properties: {
    model: { format: 'OpenAI', name: modelName, version: modelVersion }
    raiPolicyName: 'Microsoft.DefaultV2'
  }
}

// ---- roles for the workload identity

resource tableRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storage
  name: guid(storage.id, identity.id, roles.tableContributor)
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.tableContributor)
  }
}

resource blobRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storage
  name: guid(storage.id, identity.id, roles.blobContributor)
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.blobContributor)
  }
}

resource queueRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storage
  name: guid(storage.id, identity.id, roles.queueContributor)
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.queueContributor)
  }
}

resource vaultRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: vault
  name: guid(vault.id, identity.id, roles.vaultSecretsUser)
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.vaultSecretsUser)
  }
}

resource pullRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: registry
  name: guid(registry.id, identity.id, roles.acrPull)
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.acrPull)
  }
}

resource modelRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (deployModel) {
  scope: ai
  name: guid(ai.id, identity.id, roles.openAiUser)
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.openAiUser)
  }
}

// ---- roles for the person who operates it (writing secrets, issuing device tokens, reading data)

resource adminVault 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(adminPrincipalId)) {
  scope: vault
  name: guid(vault.id, adminPrincipalId, roles.vaultSecretsOfficer)
  properties: {
    principalId: adminPrincipalId
    principalType: 'User'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.vaultSecretsOfficer)
  }
}

resource adminData 'Microsoft.Authorization/roleAssignments@2022-04-01' = [for role in ['tableContributor', 'blobContributor', 'queueContributor']: if (!empty(adminPrincipalId)) {
  scope: storage
  name: guid(storage.id, adminPrincipalId, roles[role])
  properties: {
    principalId: adminPrincipalId
    principalType: 'User'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles[role])
  }
}]

resource adminModel 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(adminPrincipalId) && deployModel) {
  scope: ai
  name: guid(ai.id, adminPrincipalId, roles.openAiUser)
  properties: {
    principalId: adminPrincipalId
    principalType: 'User'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.openAiUser)
  }
}

// ---- container apps

resource environment 'Microsoft.App/managedEnvironments@2025-01-01' = if (createEnvironment) {
  name: '${prefix}-env-${environmentName}'
  location: location
  tags: tags
  properties: {
    // Stated explicitly: without it the platform may create an "express" environment, which refuses Key Vault secret references.
    workloadProfiles: [
      { name: 'Consumption', workloadProfileType: 'Consumption' }
    ]
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: createEnvironment ? logs!.properties.customerId : ''
        sharedKey: createEnvironment ? logs!.listKeys().primarySharedKey : ''

      }
    }
  }
}

var environmentId = createEnvironment ? environment!.id : existingEnvironmentId
var defaultDomain = createEnvironment ? environment!.properties.defaultDomain : reference(existingEnvironmentId, '2025-01-01').defaultDomain
var apiHost = '${apiName}.${defaultDomain}'
var allowedHosts = empty(extraAllowedHosts) ? apiHost : '${apiHost},${extraAllowedHosts}'
var visionEndpoint = deployModel ? 'https://${aiName}.openai.azure.com' : externalVisionEndpoint
var visionDeployment = deployModel ? modelName : externalVisionDeployment

var baseEnv = [
  { name: 'SNN_STORAGE', value: 'azure' }
  { name: 'SNN_AZURE_ACCOUNT', value: storage.name }
  { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId } // selects the user-assigned identity for storage and for the model
  { name: 'SNN_RUNTIME', value: runtime }
  { name: 'SNN_ALLOW_DEMO_RUNTIME', value: allowDemoRuntime ? '1' : '0' }
  { name: 'SNN_MANIFEST_PATH', value: manifestPath }
  { name: 'SNN_POLICY', value: policy }
  { name: 'SNN_ALLOW_LIVE', value: allowLive ? '1' : '0' }
  { name: 'SNN_OPERATOR_USERNAME', value: operatorUsername }
  { name: 'SNN_KEYVAULT_URL', value: 'https://${vaultName}${az.environment().suffixes.keyvaultDns}' }
  { name: 'SNN_OPERATOR_PASSWORD_HASH', value: 'keyvault:operator-password-hash' } // resolved by the process, see keyvault.py
  { name: 'SNN_ALLOWED_HOSTS', value: allowedHosts }
  { name: 'SNN_TRUSTED_PROXIES', value: '1' } // the Container Apps ingress is one hop
  { name: 'SNN_VISION_ENDPOINT', value: visionEndpoint }
  { name: 'SNN_VISION_DEPLOYMENT', value: visionDeployment }
  { name: 'SNN_VISION_FAMILY', value: visionFamily }
]

var mailEnv = mailEnabled ? [
  { name: 'SNN_SMTP_HOST', value: smtpHost }
  { name: 'SNN_SMTP_PORT', value: string(smtpPort) }
  { name: 'SNN_SMTP_USER', value: 'keyvault:smtp-user' }
  { name: 'SNN_SMTP_PASSWORD', value: 'keyvault:smtp-password' }
  { name: 'SNN_SMTP_FROM', value: 'keyvault:smtp-from' }
  { name: 'SNN_ALERT_RECIPIENTS', value: 'keyvault:alert-recipients' }
] : []

var sharedEnv = concat(baseEnv, mailEnv)
var sharedIdentity = {
  type: 'UserAssigned'
  userAssignedIdentities: { '${identity.id}': {} }
}
var registries = [
  {
    server: registry.properties.loginServer
    identity: identity.id
  }
]
var image = '${registry.properties.loginServer}/${imageName}'

resource api 'Microsoft.App/containerApps@2025-01-01' = if (deployApps) {
  name: apiName
  location: location
  tags: tags
  identity: sharedIdentity
  dependsOn: [vaultRole, tableRole, blobRole, queueRole, pullRole, modelRole]
  properties: {
    managedEnvironmentId: environmentId
    configuration: {
      activeRevisionsMode: 'Single'
      registries: registries
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
      }
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          args: ['api']
          env: sharedEnv
          resources: { cpu: json('0.25'), memory: '0.5Gi' }
          probes: [
            { type: 'Liveness', httpGet: { path: '/healthz', port: 8000 }, initialDelaySeconds: 10, periodSeconds: 20, failureThreshold: 3 }
            { type: 'Readiness', httpGet: { path: '/healthz', port: 8000 }, initialDelaySeconds: 5, periodSeconds: 10 }
          ]
        }
      ]
      scale: {
        minReplicas: apiMinReplicas
        maxReplicas: apiMaxReplicas
        rules: [
          { name: 'http', http: { metadata: { concurrentRequests: '50' } } }
        ]
      }
    }
  }
}

resource worker 'Microsoft.App/containerApps@2025-01-01' = if (deployApps) {
  name: workerName
  location: location
  tags: tags
  identity: sharedIdentity
  dependsOn: [vaultRole, tableRole, blobRole, queueRole, pullRole, modelRole]
  properties: {
    managedEnvironmentId: environmentId
    configuration: {
      activeRevisionsMode: 'Single'
      registries: registries
    }
    template: {
      containers: [
        {
          name: 'worker'
          image: image
          args: ['worker']
          env: sharedEnv
          resources: { cpu: json('0.25'), memory: '0.5Gi' }
        }
      ]
      scale: {
        minReplicas: 1 // the outbox reconciler runs in the worker, so at least one must always be alive
        maxReplicas: scaleWorkerOnQueue ? workerMaxReplicas : 1
        rules: scaleWorkerOnQueue ? [
          {
            name: 'vision-queue'
            custom: {
              type: 'azure-queue'
              metadata: {
                accountName: storage.name
                queueName: 'vision-jobs'
                queueLength: '5'
              }
              identity: identity.id // KEDA reads the queue length with the managed identity, no key
            }
          }
        ] : []
      }
    }
  }
}

output resourceGroup string = resourceGroup().name
output apiHost string = apiHost
output registryName string = registry.name
output registryServer string = registry.properties.loginServer
output vaultName string = vault.name
output storageAccount string = storage.name
output identityClientId string = identity.properties.clientId
output identityPrincipalId string = identity.properties.principalId
output visionEndpoint string = visionEndpoint
output visionDeployment string = visionDeployment
