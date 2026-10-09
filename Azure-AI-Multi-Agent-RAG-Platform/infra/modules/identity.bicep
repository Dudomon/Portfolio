// User-assigned managed identity for the API. A user-assigned identity (rather than a
// system-assigned one) exists before the container app, so role assignments and the
// registry pull permission are in place by the time the first revision starts.

param location string
param name string
param tags object

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: name
  location: location
  tags: tags
}

output id string = identity.id
output principalId string = identity.properties.principalId
output clientId string = identity.properties.clientId
