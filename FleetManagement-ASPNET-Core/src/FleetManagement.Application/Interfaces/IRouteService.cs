using FleetManagement.Application.DTOs;
using FleetManagement.Domain.Enums;

namespace FleetManagement.Application.Interfaces;

public interface IRouteService
{
    Task<RouteDto?> GetByIdAsync(Guid id, CancellationToken cancellationToken = default);
    Task<IEnumerable<RouteDto>> GetAllAsync(CancellationToken cancellationToken = default);
    Task<IEnumerable<RouteDto>> GetActiveRoutesAsync(CancellationToken cancellationToken = default);
    Task<IEnumerable<RouteDto>> GetRoutesByStatusAsync(RouteStatus status, CancellationToken cancellationToken = default);
    Task<RouteDto> CreateAsync(CreateRouteDto dto, CancellationToken cancellationToken = default);
    Task<RouteDto> UpdateStatusAsync(Guid id, RouteStatus status, CancellationToken cancellationToken = default);
    Task DeleteAsync(Guid id, CancellationToken cancellationToken = default);
}
