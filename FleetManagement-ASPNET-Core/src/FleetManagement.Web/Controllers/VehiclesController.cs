using FleetManagement.Application.DTOs;
using FleetManagement.Application.Interfaces;
using FleetManagement.Domain.Enums;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace FleetManagement.Web.Controllers;

[Authorize]
[Route("api/[controller]")]
[ApiController]
public class VehiclesController : ControllerBase
{
    private readonly IVehicleService _vehicleService;
    private readonly ILogger<VehiclesController> _logger;

    public VehiclesController(IVehicleService vehicleService, ILogger<VehiclesController> logger)
    {
        _vehicleService = vehicleService;
        _logger = logger;
    }

    [HttpGet]
    [ProducesResponseType(typeof(IEnumerable<VehicleDto>), StatusCodes.Status200OK)]
    public async Task<ActionResult<IEnumerable<VehicleDto>>> GetAll(CancellationToken cancellationToken)
    {
        var vehicles = await _vehicleService.GetAllAsync(cancellationToken);
        return Ok(vehicles);
    }

    [HttpGet("{id:guid}")]
    [ProducesResponseType(typeof(VehicleDto), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<ActionResult<VehicleDto>> GetById(Guid id, CancellationToken cancellationToken)
    {
        var vehicle = await _vehicleService.GetByIdAsync(id, cancellationToken);
        if (vehicle == null)
        {
            return NotFound();
        }

        return Ok(vehicle);
    }

    [HttpGet("available")]
    [ProducesResponseType(typeof(IEnumerable<VehicleDto>), StatusCodes.Status200OK)]
    public async Task<ActionResult<IEnumerable<VehicleDto>>> GetAvailable(CancellationToken cancellationToken)
    {
        var vehicles = await _vehicleService.GetAvailableVehiclesAsync(cancellationToken);
        return Ok(vehicles);
    }

    [HttpGet("status/{status}")]
    [ProducesResponseType(typeof(IEnumerable<VehicleDto>), StatusCodes.Status200OK)]
    public async Task<ActionResult<IEnumerable<VehicleDto>>> GetByStatus(
        VehicleStatus status,
        CancellationToken cancellationToken)
    {
        var vehicles = await _vehicleService.GetVehiclesByStatusAsync(status, cancellationToken);
        return Ok(vehicles);
    }

    [HttpGet("plate/{plateNumber}")]
    [ProducesResponseType(typeof(VehicleDto), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<ActionResult<VehicleDto>> GetByPlateNumber(
        string plateNumber,
        CancellationToken cancellationToken)
    {
        var vehicle = await _vehicleService.GetByPlateNumberAsync(plateNumber, cancellationToken);
        if (vehicle == null)
        {
            return NotFound();
        }

        return Ok(vehicle);
    }

    [HttpGet("maintenance-needed")]
    [ProducesResponseType(typeof(IEnumerable<VehicleDto>), StatusCodes.Status200OK)]
    public async Task<ActionResult<IEnumerable<VehicleDto>>> GetMaintenanceNeeded(CancellationToken cancellationToken)
    {
        var vehicles = await _vehicleService.GetVehiclesNeedingMaintenanceAsync(cancellationToken);
        return Ok(vehicles);
    }

    [HttpPost]
    [ProducesResponseType(typeof(VehicleDto), StatusCodes.Status201Created)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<ActionResult<VehicleDto>> Create(
        [FromBody] CreateVehicleDto dto,
        CancellationToken cancellationToken)
    {
        try
        {
            var vehicle = await _vehicleService.CreateAsync(dto, cancellationToken);
            return CreatedAtAction(nameof(GetById), new { id = vehicle.Id }, vehicle);
        }
        catch (InvalidOperationException ex)
        {
            _logger.LogWarning(ex, "Failed to create vehicle");
            return BadRequest(new { error = ex.Message });
        }
    }

    [HttpPut("{id:guid}")]
    [ProducesResponseType(typeof(VehicleDto), StatusCodes.Status200OK)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<ActionResult<VehicleDto>> Update(
        Guid id,
        [FromBody] UpdateVehicleDto dto,
        CancellationToken cancellationToken)
    {
        if (id != dto.Id)
        {
            return BadRequest("ID mismatch");
        }

        try
        {
            var vehicle = await _vehicleService.UpdateAsync(dto, cancellationToken);
            return Ok(vehicle);
        }
        catch (KeyNotFoundException ex)
        {
            _logger.LogWarning(ex, "Vehicle not found for update");
            return NotFound(new { error = ex.Message });
        }
    }

    [HttpDelete("{id:guid}")]
    [ProducesResponseType(StatusCodes.Status204NoContent)]
    [ProducesResponseType(StatusCodes.Status404NotFound)]
    public async Task<IActionResult> Delete(Guid id, CancellationToken cancellationToken)
    {
        try
        {
            await _vehicleService.DeleteAsync(id, cancellationToken);
            return NoContent();
        }
        catch (KeyNotFoundException ex)
        {
            _logger.LogWarning(ex, "Vehicle not found for deletion");
            return NotFound(new { error = ex.Message });
        }
    }
}
