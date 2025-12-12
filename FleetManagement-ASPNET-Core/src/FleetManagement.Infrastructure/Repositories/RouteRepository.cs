using FleetManagement.Domain.Entities;
using FleetManagement.Domain.Enums;
using FleetManagement.Domain.Interfaces;
using FleetManagement.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;

namespace FleetManagement.Infrastructure.Repositories;

public class RouteRepository : Repository<Route>, IRouteRepository
{
    public RouteRepository(FleetManagementDbContext context) : base(context)
    {
    }

    public async Task<IEnumerable<Route>> GetActiveRoutesAsync(CancellationToken cancellationToken = default)
    {
        return await _dbSet
            .Where(r => r.Status == RouteStatus.InProgress || r.Status == RouteStatus.Scheduled)
            .Include(r => r.Vehicle)
            .Include(r => r.Driver)
            .ToListAsync(cancellationToken);
    }

    public async Task<IEnumerable<Route>> GetRoutesByStatusAsync(
        RouteStatus status,
        CancellationToken cancellationToken = default)
    {
        return await _dbSet
            .Where(r => r.Status == status)
            .Include(r => r.Vehicle)
            .Include(r => r.Driver)
            .ToListAsync(cancellationToken);
    }

    public async Task<IEnumerable<Route>> GetRoutesByDriverAsync(
        Guid driverId,
        CancellationToken cancellationToken = default)
    {
        return await _dbSet
            .Where(r => r.DriverId == driverId)
            .Include(r => r.Vehicle)
            .Include(r => r.Shipments)
            .OrderByDescending(r => r.ScheduledDepartureDate)
            .ToListAsync(cancellationToken);
    }

    public async Task<IEnumerable<Route>> GetRoutesByVehicleAsync(
        Guid vehicleId,
        CancellationToken cancellationToken = default)
    {
        return await _dbSet
            .Where(r => r.VehicleId == vehicleId)
            .Include(r => r.Driver)
            .Include(r => r.Shipments)
            .OrderByDescending(r => r.ScheduledDepartureDate)
            .ToListAsync(cancellationToken);
    }

    public async Task<Route?> GetRouteWithDetailsAsync(Guid id, CancellationToken cancellationToken = default)
    {
        return await _dbSet
            .Include(r => r.Vehicle)
            .Include(r => r.Driver)
            .Include(r => r.Shipments)
            .Include(r => r.Checkpoints.OrderBy(c => c.SequenceNumber))
            .FirstOrDefaultAsync(r => r.Id == id, cancellationToken);
    }
}
