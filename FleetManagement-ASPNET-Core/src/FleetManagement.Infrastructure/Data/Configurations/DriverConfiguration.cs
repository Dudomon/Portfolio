using FleetManagement.Domain.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;

namespace FleetManagement.Infrastructure.Data.Configurations;

public class DriverConfiguration : IEntityTypeConfiguration<Driver>
{
    public void Configure(EntityTypeBuilder<Driver> builder)
    {
        builder.ToTable("Drivers");

        builder.HasKey(d => d.Id);

        builder.Property(d => d.FirstName)
            .IsRequired()
            .HasMaxLength(100);

        builder.Property(d => d.LastName)
            .IsRequired()
            .HasMaxLength(100);

        builder.Property(d => d.Email)
            .IsRequired()
            .HasMaxLength(255);

        builder.HasIndex(d => d.Email)
            .IsUnique();

        builder.Property(d => d.PhoneNumber)
            .IsRequired()
            .HasMaxLength(20);

        builder.Property(d => d.LicenseNumber)
            .IsRequired()
            .HasMaxLength(50);

        builder.HasIndex(d => d.LicenseNumber)
            .IsUnique();

        builder.Property(d => d.Address)
            .IsRequired()
            .HasMaxLength(255);

        builder.Property(d => d.City)
            .IsRequired()
            .HasMaxLength(100);

        builder.Property(d => d.State)
            .IsRequired()
            .HasMaxLength(50);

        builder.Property(d => d.ZipCode)
            .IsRequired()
            .HasMaxLength(20);

        builder.HasMany(d => d.Routes)
            .WithOne(r => r.Driver)
            .HasForeignKey(r => r.DriverId)
            .OnDelete(DeleteBehavior.Restrict);

        builder.HasMany(d => d.Documents)
            .WithOne(dd => dd.Driver)
            .HasForeignKey(dd => dd.DriverId)
            .OnDelete(DeleteBehavior.Cascade);
    }
}
