// Container registry, Container Apps environment and the API container app.

param location string
param tags object
param registryName string
param environmentName string
param appName string
param logAnalyticsWorkspaceName string
param identityId string
param identityPrincipalId string
param identityClientId string

@description('Image to run. The first deployment uses a public placeholder because the new registry is empty.')
param containerImage string

@secure()
param appInsightsConnectionString string

@description('Plain (non-secret) environment variables for the API.')
param appSettings object

@description('Expose the API publicly. Put Entra ID authentication (Easy Auth) or a gateway in front when true.')
param externalIngress bool = true

param minReplicas int = 1
param maxReplicas int = 5

var acrPullRoleId = '7f951dda-4ed3-4680-a7ca-43fe172d538d'
var usesPrivateRegistry = startsWith(containerImage, '${registryName}.azurecr.io/')

resource workspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' existing = {
  name: logAnalyticsWorkspaceName
}

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: registryName
  location: location
  tags: tags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(registry.id, identityPrincipalId, acrPullRoleId)
  scope: registry
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', acrPullRoleId)
    principalId: identityPrincipalId
    principalType: 'ServicePrincipal'
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: workspace.properties.customerId
        sharedKey: workspace.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  tags: tags
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identityId}': {}
    }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: externalIngress
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      registries: usesPrivateRegistry
        ? [
            {
              server: registry.properties.loginServer
              identity: identityId
            }
          ]
        : []
      secrets: [
        {
          name: 'appinsights-connection-string'
          value: appInsightsConnectionString
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: containerImage
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: concat(
            [
              {
                // Makes DefaultAzureCredential select this user-assigned identity.
                name: 'AZURE_CLIENT_ID'
                value: identityClientId
              }
              {
                name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
                secretRef: 'appinsights-connection-string'
              }
            ],
            map(items(appSettings), setting => { name: setting.key, value: string(setting.value) })
          )
          probes: [
            {
              type: 'Liveness'
              httpGet: {
                path: '/healthz'
                port: 8000
              }
              periodSeconds: 15
            }
            {
              type: 'Readiness'
              httpGet: {
                path: '/healthz'
                port: 8000
              }
              initialDelaySeconds: 3
              periodSeconds: 10
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            // Requests are long-lived (streaming, several model calls), so scale on
            // concurrency rather than CPU.
            name: 'http-concurrency'
            http: {
              metadata: {
                concurrentRequests: '20'
              }
            }
          }
        ]
      }
    }
  }
  dependsOn: [
    acrPull
  ]
}

output registryLoginServer string = registry.properties.loginServer
output appFqdn string = app.properties.configuration.ingress.fqdn
