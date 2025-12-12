using FleetManagement.Domain.Entities;
using FleetManagement.Domain.Enums;

namespace FleetManagement.Domain.Interfaces;

public interface IRouteRepository : IRepository<Route>
{
    Task<IEnumerable<Route>> GetActiveRoutesAsync(CancellationToken cancellationToken = default);
    Task<IEnumerable<Route>> GetRoutesByStatusAsync(RouteStatus status, CancellationToken cancellationToken = default);
    Task<IEnumerable<Route>> GetRoutesByDriverAsync(Guid driverId, CancellationToken cancellationToken = default);
    Task<IEnumerable<Route>> GetRoutesByVehicleAsync(Guid vehicleId, CancellationToken cancellationToken = default);
    Task<Route?> GetRouteWithDetailsAsync(Guid id, CancellationToken cancellationToken = default);
}
