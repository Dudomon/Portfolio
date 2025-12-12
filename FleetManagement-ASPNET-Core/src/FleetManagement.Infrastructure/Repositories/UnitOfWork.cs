using FleetManagement.Domain.Entities;
using FleetManagement.Domain.Interfaces;
using FleetManagement.Infrastructure.Data;
using Microsoft.EntityFrameworkCore.Storage;

namespace FleetManagement.Infrastructure.Repositories;

public class UnitOfWork : IUnitOfWork
{
    private readonly FleetManagementDbContext _context;
    private IDbContextTransaction? _transaction;

    public UnitOfWork(FleetManagementDbContext context)
    {
        _context = context;
        Vehicles = new VehicleRepository(context);
        Drivers = new DriverRepository(context);
        Routes = new RouteRepository(context);
        Shipments = new Repository<Shipment>(context);
        Maintenances = new Repository<Maintenance>(context);
    }

    public IVehicleRepository Vehicles { get; private set; }
    public IDriverRepository Drivers { get; private set; }
    public IRouteRepository Routes { get; private set; }
    public IRepository<Shipment> Shipments { get; private set; }
    public IRepository<Maintenance> Maintenances { get; private set; }

    public async Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
    {
        return await _context.SaveChangesAsync(cancellationToken);
    }

    public async Task BeginTransactionAsync(CancellationToken cancellationToken = default)
    {
        _transaction = await _context.Database.BeginTransactionAsync(cancellationToken);
    }

    public async Task CommitTransactionAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            await SaveChangesAsync(cancellationToken);
            if (_transaction != null)
            {
                await _transaction.CommitAsync(cancellationToken);
            }
        }
        catch
        {
            await RollbackTransactionAsync(cancellationToken);
            throw;
        }
        finally
        {
            if (_transaction != null)
            {
                await _transaction.DisposeAsync();
                _transaction = null;
            }
        }
    }

    public async Task RollbackTransactionAsync(CancellationToken cancellationToken = default)
    {
        if (_transaction != null)
        {
            await _transaction.RollbackAsync(cancellationToken);
            await _transaction.DisposeAsync();
            _transaction = null;
        }
    }

    public void Dispose()
    {
        _transaction?.Dispose();
        _context.Dispose();
    }
}
