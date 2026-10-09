// Least-privilege data-plane role assignments, kept in one module so the full RBAC
// surface can be reviewed in one place.
//
// API identity:  OpenAI User (inference only) + Search Index Data Reader (query only).
// Deployer:      OpenAI User + Search Index Data Contributor + Search Service Contributor,
//                which is what `rag-ingest` needs to create the index and upload chunks.

param openAiName string
param searchName string
param apiPrincipalId string

@description('Object id of the user or pipeline principal that runs ingestion. Empty to skip.')
param deployerPrincipalId string = ''

@allowed([
  'User'
  'ServicePrincipal'
  'Group'
])
param deployerPrincipalType string = 'User'

var roles = {
  openAiUser: '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
  searchIndexDataReader: '1407120a-92aa-4202-b7e9-c0e197c71c8f'
  searchIndexDataContributor: '8ebe5a00-799e-43f5-93ac-243d3dce84a7'
  searchServiceContributor: '7ca78c08-252a-4471-8644-bb5ff32d4ba0'
}

resource openAi 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: openAiName
}

resource search 'Microsoft.Search/searchServices@2023-11-01' existing = {
  name: searchName
}

resource apiOpenAiUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(openAi.id, apiPrincipalId, roles.openAiUser)
  scope: openAi
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.openAiUser)
    principalId: apiPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource apiSearchReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(search.id, apiPrincipalId, roles.searchIndexDataReader)
  scope: search
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.searchIndexDataReader)
    principalId: apiPrincipalId
    principalType: 'ServicePrincipal'
  }
}

var deployerSearchRoles = empty(deployerPrincipalId)
  ? []
  : [
      roles.searchIndexDataContributor
      roles.searchServiceContributor
    ]

resource deployerSearch 'Microsoft.Authorization/roleAssignments@2022-04-01' = [
  for role in deployerSearchRoles: {
    name: guid(search.id, deployerPrincipalId, role)
    scope: search
    properties: {
      roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', role)
      principalId: deployerPrincipalId
      principalType: deployerPrincipalType
    }
  }
]

resource deployerOpenAiUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(deployerPrincipalId)) {
  name: guid(openAi.id, deployerPrincipalId, roles.openAiUser)
  scope: openAi
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.openAiUser)
    principalId: deployerPrincipalId
    principalType: deployerPrincipalType
  }
}
