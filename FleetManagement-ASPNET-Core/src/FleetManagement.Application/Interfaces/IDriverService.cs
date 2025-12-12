using FleetManagement.Application.DTOs;

namespace FleetManagement.Application.Interfaces;

public interface IDriverService
{
    Task<DriverDto?> GetByIdAsync(Guid id, CancellationToken cancellationToken = default);
    Task<IEnumerable<DriverDto>> GetAllAsync(CancellationToken cancellationToken = default);
    Task<IEnumerable<DriverDto>> GetActiveDriversAsync(CancellationToken cancellationToken = default);
    Task<DriverDto> CreateAsync(CreateDriverDto dto, CancellationToken cancellationToken = default);
    Task<DriverDto> UpdateAsync(DriverDto dto, CancellationToken cancellationToken = default);
    Task DeleteAsync(Guid id, CancellationToken cancellationToken = default);
    Task<DriverDto?> GetByLicenseNumberAsync(string licenseNumber, CancellationToken cancellationToken = default);
}
