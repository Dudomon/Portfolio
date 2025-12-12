using FleetManagement.Domain.Entities;
using FleetManagement.Domain.Enums;

namespace FleetManagement.Domain.Interfaces;

public interface IDriverRepository : IRepository<Driver>
{
    Task<IEnumerable<Driver>> GetActiveDriversAsync(CancellationToken cancellationToken = default);
    Task<Driver?> GetDriverByLicenseNumberAsync(string licenseNumber, CancellationToken cancellationToken = default);
    Task<IEnumerable<Driver>> GetDriversWithExpiringLicensesAsync(int daysThreshold, CancellationToken cancellationToken = default);
}
