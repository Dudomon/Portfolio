using AutoMapper;
using FleetManagement.Application.DTOs;
using FleetManagement.Application.Interfaces;
using FleetManagement.Domain.Entities;
using FleetManagement.Domain.Enums;
using FleetManagement.Domain.Interfaces;
using Microsoft.Extensions.Logging;

namespace FleetManagement.Application.Services;

public class VehicleService : IVehicleService
{
    private readonly IUnitOfWork _unitOfWork;
    private readonly IMapper _mapper;
    private readonly ILogger<VehicleService> _logger;

    public VehicleService(IUnitOfWork unitOfWork, IMapper mapper, ILogger<VehicleService> logger)
    {
        _unitOfWork = unitOfWork;
        _mapper = mapper;
        _logger = logger;
    }

    public async Task<VehicleDto?> GetByIdAsync(Guid id, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Retrieving vehicle with ID: {VehicleId}", id);
        var vehicle = await _unitOfWork.Vehicles.GetByIdAsync(id, cancellationToken);
        return vehicle == null ? null : _mapper.Map<VehicleDto>(vehicle);
    }

    public async Task<IEnumerable<VehicleDto>> GetAllAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Retrieving all vehicles");
        var vehicles = await _unitOfWork.Vehicles.GetAllAsync(cancellationToken);
        return _mapper.Map<IEnumerable<VehicleDto>>(vehicles);
    }

    public async Task<IEnumerable<VehicleDto>> GetAvailableVehiclesAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Retrieving available vehicles");
        var vehicles = await _unitOfWork.Vehicles.GetAvailableVehiclesAsync(cancellationToken);
        return _mapper.Map<IEnumerable<VehicleDto>>(vehicles);
    }

    public async Task<IEnumerable<VehicleDto>> GetVehiclesByStatusAsync(
        VehicleStatus status,
        CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Retrieving vehicles with status: {Status}", status);
        var vehicles = await _unitOfWork.Vehicles.GetVehiclesByStatusAsync(status, cancellationToken);
        return _mapper.Map<IEnumerable<VehicleDto>>(vehicles);
    }

    public async Task<VehicleDto> CreateAsync(CreateVehicleDto dto, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Creating new vehicle with plate number: {PlateNumber}", dto.PlateNumber);

        var existingVehicle = await _unitOfWork.Vehicles.GetVehicleByPlateNumberAsync(dto.PlateNumber, cancellationToken);
        if (existingVehicle != null)
        {
            _logger.LogWarning("Vehicle with plate number {PlateNumber} already exists", dto.PlateNumber);
            throw new InvalidOperationException($"Vehicle with plate number {dto.PlateNumber} already exists");
        }

        var vehicle = _mapper.Map<Vehicle>(dto);
        vehicle.Status = VehicleStatus.Available;

        await _unitOfWork.Vehicles.AddAsync(vehicle, cancellationToken);
        await _unitOfWork.SaveChangesAsync(cancellationToken);

        _logger.LogInformation("Vehicle created successfully with ID: {VehicleId}", vehicle.Id);
        return _mapper.Map<VehicleDto>(vehicle);
    }

    public async Task<VehicleDto> UpdateAsync(UpdateVehicleDto dto, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Updating vehicle with ID: {VehicleId}", dto.Id);

        var vehicle = await _unitOfWork.Vehicles.GetByIdAsync(dto.Id, cancellationToken);
        if (vehicle == null)
        {
            _logger.LogWarning("Vehicle with ID {VehicleId} not found", dto.Id);
            throw new KeyNotFoundException($"Vehicle with ID {dto.Id} not found");
        }

        _mapper.Map(dto, vehicle);
        await _unitOfWork.Vehicles.UpdateAsync(vehicle, cancellationToken);
        await _unitOfWork.SaveChangesAsync(cancellationToken);

        _logger.LogInformation("Vehicle updated successfully with ID: {VehicleId}", vehicle.Id);
        return _mapper.Map<VehicleDto>(vehicle);
    }

    public async Task DeleteAsync(Guid id, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Deleting vehicle with ID: {VehicleId}", id);

        var vehicle = await _unitOfWork.Vehicles.GetByIdAsync(id, cancellationToken);
        if (vehicle == null)
        {
            _logger.LogWarning("Vehicle with ID {VehicleId} not found", id);
            throw new KeyNotFoundException($"Vehicle with ID {id} not found");
        }

        await _unitOfWork.Vehicles.DeleteAsync(id, cancellationToken);
        await _unitOfWork.SaveChangesAsync(cancellationToken);

        _logger.LogInformation("Vehicle deleted successfully with ID: {VehicleId}", id);
    }

    public async Task<VehicleDto?> GetByPlateNumberAsync(string plateNumber, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Retrieving vehicle with plate number: {PlateNumber}", plateNumber);
        var vehicle = await _unitOfWork.Vehicles.GetVehicleByPlateNumberAsync(plateNumber, cancellationToken);
        return vehicle == null ? null : _mapper.Map<VehicleDto>(vehicle);
    }

    public async Task<IEnumerable<VehicleDto>> GetVehiclesNeedingMaintenanceAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("Retrieving vehicles needing maintenance");
        var vehicles = await _unitOfWork.Vehicles.GetVehiclesNeedingMaintenanceAsync(cancellationToken);
        return _mapper.Map<IEnumerable<VehicleDto>>(vehicles);
    }
}
