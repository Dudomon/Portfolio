using FleetManagement.Domain.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace FleetManagement.Infrastructure.Data.Configurations;

public class RouteConfiguration : IEntityTypeConfiguration<Route>
{
    public void Configure(EntityTypeBuilder<Route> builder)
    {
        builder.ToTable("Routes");

        builder.HasKey(r => r.Id);

        builder.Property(r => r.RouteNumber)
            .IsRequired()
            .HasMaxLength(50);

        builder.HasIndex(r => r.RouteNumber)
            .IsUnique();

        builder.Property(r => r.OriginAddress)
            .IsRequired()
            .HasMaxLength(255);

        builder.Property(r => r.DestinationAddress)
            .IsRequired()
            .HasMaxLength(255);

        builder.Property(r => r.OriginLatitude)
            .HasPrecision(10, 7);

        builder.Property(r => r.OriginLongitude)
            .HasPrecision(10, 7);

        builder.Property(r => r.DestinationLatitude)
            .HasPrecision(10, 7);

        builder.Property(r => r.DestinationLongitude)
            .HasPrecision(10, 7);

        builder.Property(r => r.EstimatedDistance)
            .HasPrecision(18, 2);

        builder.Property(r => r.ActualDistance)
            .HasPrecision(18, 2);

        builder.Property(r => r.EstimatedFuelCost)
            .HasPrecision(18, 2);

        builder.Property(r => r.ActualFuelCost)
            .HasPrecision(18, 2);

        builder.Property(r => r.TollCost)
            .HasPrecision(18, 2);

        builder.HasOne(r => r.Vehicle)
            .WithMany(v => v.Routes)
            .HasForeignKey(r => r.VehicleId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasOne(r => r.Driver)
            .WithMany(d => d.Routes)
            .HasForeignKey(r => r.DriverId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasMany(r => r.Shipments)
            .WithOne(s => s.Route)
            .HasForeignKey(s => s.RouteId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasMany(r => r.Checkpoints)
            .WithOne(c => c.Route)
            .HasForeignKey(c => c.RouteId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}
