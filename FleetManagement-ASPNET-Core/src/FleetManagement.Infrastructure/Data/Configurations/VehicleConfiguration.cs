using FleetManagement.Domain.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace FleetManagement.Infrastructure.Data.Configurations;

public class VehicleConfiguration : IEntityTypeConfiguration<Vehicle>
{
    public void Configure(EntityTypeBuilder<Vehicle> builder)
    {
        builder.ToTable("Vehicles");

        builder.HasKey(v => v.Id);

        builder.Property(v => v.PlateNumber)
            .IsRequired()
            .HasMaxLength(20);

        builder.HasIndex(v => v.PlateNumber)
            .IsUnique();

        builder.Property(v => v.Brand)
            .IsRequired()
            .HasMaxLength(100);

        builder.Property(v => v.Model)
            .IsRequired()
            .HasMaxLength(100);

        builder.Property(v => v.VinNumber)
            .IsRequired()
            .HasMaxLength(50);

        builder.HasIndex(v => v.VinNumber)
            .IsUnique();

        builder.Property(v => v.LoadCapacity)
            .HasPrecision(18, 2);

        builder.Property(v => v.CurrentMileage)
            .HasPrecision(18, 2);

        builder.Property(v => v.FuelConsumption)
            .HasPrecision(18, 2);

        builder.HasOne(v => v.CurrentDriver)
            .WithMany()
            .HasForeignKey(v => v.CurrentDriverId)
            .OnDelete(DeleteBehavior.SetNull);

        builder.HasMany(v => v.Maintenances)
            .WithOne(m => m.Vehicle)
            .HasForeignKey(m => m.VehicleId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasMany(v => v.Routes)
            .WithOne(r => r.Vehicle)
            .HasForeignKey(r => r.VehicleId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasMany(v => v.Documents)
            .WithOne(d => d.Vehicle)
            .HasForeignKey(d => d.VehicleId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}
