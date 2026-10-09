// Azure OpenAI account with model deployments. Key-based auth is disabled: every
// caller authenticates with Entra ID and needs the "Cognitive Services OpenAI User" role.

param location string
param name string
param tags object

@description('Model deployments: name, model, version, sku (e.g. GlobalStandard) and capacity in thousands of TPM.')
param deployments array

resource account 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: name
  location: location
  tags: tags
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: name
    disableLocalAuth: true
    publicNetworkAccess: 'Enabled'
  }
}

// Deployments on one account must be created sequentially; parallel PUTs conflict.
@batchSize(1)
resource modelDeployments 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = [
  for deployment in deployments: {
    parent: account
    name: deployment.name
    sku: {
      name: deployment.sku
      capacity: deployment.capacity
    }
    properties: {
      model: {
        format: 'OpenAI'
        name: deployment.model
        version: deployment.version
      }
      // Default content filtering (including Prompt Shields for jailbreaks) applies;
      // the application handles filter results explicitly.
      versionUpgradeOption: 'NoAutoUpgrade'
    }
  }
]

output name string = account.name
output endpoint string = account.properties.endpoint
