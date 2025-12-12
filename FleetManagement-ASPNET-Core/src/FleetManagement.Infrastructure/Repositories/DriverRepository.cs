using FleetManagement.Domain.Entities;
using FleetManagement.Domain.Enums;
using FleetManagement.Domain.Interfaces;
using FleetManagement.Infrastructure.Data;
using Microsoft.EntityFrameworkCore;

namespace FleetManagement.Infrastructure.Repositories;

public class DriverRepository : Repository<Driver>, IDriverRepository
{
    public DriverRepository(FleetManagementDbContext context) : base(context)
    {
    }

    public async Task<IEnumerable<Driver>> GetActiveDriversAsync(CancellationToken cancellationToken = default)
    {
        return await _dbSet
            .Where(d => d.Status == DriverStatus.Active)
            .ToListAsync(cancellationToken);
    }

    public async Task<Driver?> GetDriverByLicenseNumberAsync(
        string licenseNumber,
        CancellationToken cancellationToken = default)
    {
        return await _dbSet
            .FirstOrDefaultAsync(d => d.LicenseNumber == licenseNumber, cancellationToken);
    }

    public async Task<IEnumerable<Driver>> GetDriversWithExpiringLicensesAsync(
        int daysThreshold,
        CancellationToken cancellationToken = default)
    {
        var thresholdDate = DateTime.UtcNow.AddDays(daysThreshold);
        return await _dbSet
            .Where(d => d.LicenseExpiryDate <= thresholdDate && d.Status == DriverStatus.Active)
            .ToListAsync(cancellationToken);
    }
}
