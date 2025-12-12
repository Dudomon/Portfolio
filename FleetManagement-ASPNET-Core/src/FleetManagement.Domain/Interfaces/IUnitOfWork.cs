namespace FleetManagement.Domain.Interfaces;

public interface IUnitOfWork : IDisposable
{
    IVehicleRepository Vehicles { get; }
    IDriverRepository Drivers { get; }
    IRouteRepository Routes { get; }
    IRepository<Entities.Shipment> Shipments { get; }
    IRepository<Entities.Maintenance> Maintenances { get; }

    Task<int> SaveChangesAsync(CancellationToken cancellationToken = default);
    Task BeginTransactionAsync(CancellationToken cancellationToken = default);
    Task CommitTransactionAsync(CancellationToken cancellationToken = default);
    Task RollbackTransactionAsync(CancellationToken cancellationToken = default);
}
