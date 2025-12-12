using FleetManagement.Domain.Entities;
using Microsoft.EntityFrameworkCore;
using System.Reflection;

namespace FleetManagement.Infrastructure.Data;

public class FleetManagementDbContext : DbContext
{
    public FleetManagementDbContext(DbContextOptions<FleetManagementDbContext> options)
        : base(options)
    {
    }

    public DbSet<Vehicle> Vehicles => Set<Vehicle>();
    public DbSet<Driver> Drivers => Set<Driver>();
    public DbSet<Route> Routes => Set<Route>();
    public DbSet<Shipment> Shipments => Set<Shipment>();
    public DbSet<Maintenance> Maintenances => Set<Maintenance>();
    public DbSet<VehicleDocument> VehicleDocuments => Set<VehicleDocument>();
    public DbSet<DriverDocument> DriverDocuments => Set<DriverDocument>();
    public DbSet<RouteCheckpoint> RouteCheckpoints => Set<RouteCheckpoint>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        modelBuilder.ApplyConfigurationsFromAssembly(Assembly.GetExecutingAssembly());

        foreach (var entityType in modelBuilder.Model.GetEntityTypes())
        {
            var type = entityType.ClrType;
            if (typeof(BaseEntity).IsAssignableFrom(type))
            {
                modelBuilder.Entity(type).HasQueryFilter(
                    Expression.Lambda(
                        Expression.Equal(
                            Expression.Property(
                                Expression.Parameter(type, "e"),
                                nameof(BaseEntity.IsDeleted)
                            ),
                            Expression.Constant(false)
                        ),
                        Expression.Parameter(type, "e")
                    )
                );
            }
        }
    }

    public override Task<int> SaveChangesAsync(CancellationToken cancellationToken = default)
    {
        foreach (var entry in ChangeTracker.Entries<BaseEntity>())
        {
            switch (entry.State)
            {
                case EntityState.Added:
                    entry.Entity.CreatedAt = DateTime.UtcNow;
                    entry.Entity.Id = Guid.NewGuid();
                    break;
                case EntityState.Modified:
                    entry.Entity.UpdatedAt = DateTime.UtcNow;
                    break;
                case EntityState.Deleted:
                    entry.State = EntityState.Modified;
                    entry.Entity.IsDeleted = true;
                    entry.Entity.UpdatedAt = DateTime.UtcNow;
                    break;
            }
        }

        return base.SaveChangesAsync(cancellationToken);
    }
}
