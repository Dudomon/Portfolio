// Azure AI Search service with the semantic ranker enabled and key auth disabled.

param location string
param name string
param tags object

@allowed([
  'basic'
  'standard'
  'standard2'
])
@description('Basic is the smallest tier that supports the semantic ranker.')
param sku string = 'basic'

@minValue(1)
param replicaCount int = 1

resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: name
  location: location
  tags: tags
  sku: {
    name: sku
  }
  properties: {
    replicaCount: replicaCount
    partitionCount: 1
    hostingMode: 'default'
    semanticSearch: 'free'
    disableLocalAuth: true
    publicNetworkAccess: 'enabled'
  }
}

output name string = search.name
output endpoint string = 'https://${search.name}.search.windows.net'
