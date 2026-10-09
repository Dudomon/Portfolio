// Policy knowledge assistant: Azure OpenAI, Azure AI Search, Container Apps and
// Application Insights, wired with a user-assigned managed identity (no keys).
//
//   az deployment group create -g <rg> -f infra/main.bicep -p deployerPrincipalId=<object-id>

targetScope = 'resourceGroup'

@minLength(3)
@maxLength(10)
@description('Short lowercase prefix for resource names.')
param namePrefix string = 'kbrag'

param location string = resourceGroup().location

@description('Azure OpenAI region; model and quota availability differ per region.')
param openAiLocation string = location

param chatDeployment string = 'gpt-4o'
param utilityDeployment string = 'gpt-4o-mini'
param embeddingDeployment string = 'text-embedding-3-large'

@description('Model deployments. Capacity is in thousands of tokens per minute.')
param modelDeployments array = [
  {
    name: chatDeployment
    model: 'gpt-4o'
    version: '2024-08-06'
    sku: 'GlobalStandard'
    capacity: 50
  }
  {
    name: utilityDeployment
    model: 'gpt-4o-mini'
    version: '2024-07-18'
    sku: 'GlobalStandard'
    capacity: 100
  }
  {
    name: embeddingDeployment
    model: 'text-embedding-3-large'
    version: '1'
    sku: 'Standard'
    capacity: 50
  }
]

@allowed([
  'basic'
  'standard'
])
param searchSku string = 'basic'

param searchIndexName string = 'policies-v1'

param containerImage string = 'mcr.microsoft.com/k8se/quickstart:latest'

@description('Object id of the user or pipeline principal that runs ingestion. Empty to skip.')
param deployerPrincipalId string = ''

@allowed([
  'User'
  'ServicePrincipal'
  'Group'
])
param deployerPrincipalType string = 'User'

param externalIngress bool = true

var suffix = take(uniqueString(resourceGroup().id, namePrefix), 6)
var tags = {
  project: 'rag-platform'
  'managed-by': 'bicep'
}

module monitoring 'modules/monitoring.bicep' = {
  name: 'monitoring'
  params: {
    location: location
    workspaceName: '${namePrefix}-log-${suffix}'
    appInsightsName: '${namePrefix}-appi-${suffix}'
    tags: tags
  }
}

module identity 'modules/identity.bicep' = {
  name: 'identity'
  params: {
    location: location
    name: '${namePrefix}-api-id-${suffix}'
    tags: tags
  }
}

module openAi 'modules/openai.bicep' = {
  name: 'openai'
  params: {
    location: openAiLocation
    name: '${namePrefix}-oai-${suffix}'
    deployments: modelDeployments
    tags: tags
  }
}

module search 'modules/search.bicep' = {
  name: 'search'
  params: {
    location: location
    name: '${namePrefix}-srch-${suffix}'
    sku: searchSku
    tags: tags
  }
}

module roles 'modules/roles.bicep' = {
  name: 'roles'
  params: {
    openAiName: openAi.outputs.name
    searchName: search.outputs.name
    apiPrincipalId: identity.outputs.principalId
    deployerPrincipalId: deployerPrincipalId
    deployerPrincipalType: deployerPrincipalType
  }
}

module containerApps 'modules/containerapps.bicep' = {
  name: 'containerapps'
  params: {
    location: location
    tags: tags
    registryName: '${namePrefix}acr${suffix}'
    environmentName: '${namePrefix}-cae-${suffix}'
    appName: '${namePrefix}-api'
    logAnalyticsWorkspaceName: monitoring.outputs.workspaceName
    identityId: identity.outputs.id
    identityPrincipalId: identity.outputs.principalId
    identityClientId: identity.outputs.clientId
    containerImage: containerImage
    appInsightsConnectionString: monitoring.outputs.appInsightsConnectionString
    externalIngress: externalIngress
    appSettings: {
      AZURE_OPENAI_ENDPOINT: openAi.outputs.endpoint
      AZURE_SEARCH_ENDPOINT: search.outputs.endpoint
      CHAT_DEPLOYMENT: chatDeployment
      UTILITY_DEPLOYMENT: utilityDeployment
      JUDGE_DEPLOYMENT: chatDeployment
      EMBEDDING_DEPLOYMENT: embeddingDeployment
      SEARCH_INDEX_NAME: searchIndexName
      SERVICE_NAME: '${namePrefix}-api'
    }
  }
  dependsOn: [
    roles
  ]
}

output apiUrl string = 'https://${containerApps.outputs.appFqdn}'
output registryLoginServer string = containerApps.outputs.registryLoginServer
output azureOpenAiEndpoint string = openAi.outputs.endpoint
output azureSearchEndpoint string = search.outputs.endpoint
