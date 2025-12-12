using AutoMapper;
using FleetManagement.Application.DTOs;
using FleetManagement.Application.Mappings;
using FleetManagement.Application.Services;
using FleetManagement.Domain.Entities;
using FleetManagement.Domain.Enums;
using FleetManagement.Domain.Interfaces;
using FluentAssertions;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit;

namespace FleetManagement.UnitTests.Services;

public class VehicleServiceTests
{
    private readonly Mock<IUnitOfWork> _unitOfWorkMock;
    private readonly IMapper _mapper;
    private readonly Mock<ILogger<VehicleService>> _loggerMock;
    private readonly VehicleService _vehicleService;

    public VehicleServiceTests()
    {
        _unitOfWorkMock = new Mock<IUnitOfWork>();
        _loggerMock = new Mock<ILogger<VehicleService>>();

        var mapperConfig = new MapperConfiguration(cfg =>
        {
            cfg.AddProfile<MappingProfile>();
        });
        _mapper = mapperConfig.CreateMapper();

        _vehicleService = new VehicleService(_unitOfWorkMock.Object, _mapper, _loggerMock.Object);
    }

    [Fact]
    public async Task GetByIdAsync_ExistingVehicle_ReturnsVehicleDto()
    {
        // Arrange
        var vehicleId = Guid.NewGuid();
        var vehicle = new Vehicle
        {
            Id = vehicleId,
            PlateNumber = "ABC-1234",
            Brand = "Volvo",
            Model = "FH16",
            Year = 2023,
            Type = VehicleType.Truck,
            Status = VehicleStatus.Available,
            LoadCapacity = 25000m,
            VinNumber = "VN123456789",
            CurrentMileage = 50000m,
            FuelConsumption = 8.5m
        };

        _unitOfWorkMock.Setup(x => x.Vehicles.GetByIdAsync(vehicleId, It.IsAny<CancellationToken>()))
            .ReturnsAsync(vehicle);

        // Act
        var result = await _vehicleService.GetByIdAsync(vehicleId);

        // Assert
        result.Should().NotBeNull();
        result!.Id.Should().Be(vehicleId);
        result.PlateNumber.Should().Be("ABC-1234");
        result.Brand.Should().Be("Volvo");
        _unitOfWorkMock.Verify(x => x.Vehicles.GetByIdAsync(vehicleId, It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task GetByIdAsync_NonExistingVehicle_ReturnsNull()
    {
        // Arrange
        var vehicleId = Guid.NewGuid();
        _unitOfWorkMock.Setup(x => x.Vehicles.GetByIdAsync(vehicleId, It.IsAny<CancellationToken>()))
            .ReturnsAsync((Vehicle?)null);

        // Act
        var result = await _vehicleService.GetByIdAsync(vehicleId);

        // Assert
        result.Should().BeNull();
        _unitOfWorkMock.Verify(x => x.Vehicles.GetByIdAsync(vehicleId, It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task CreateAsync_ValidVehicle_ReturnsCreatedVehicleDto()
    {
        // Arrange
        var createDto = new CreateVehicleDto
        {
            PlateNumber = "XYZ-5678",
            Brand = "Mercedes",
            Model = "Actros",
            Year = 2024,
            Type = VehicleType.Truck,
            LoadCapacity = 30000m,
            VinNumber = "VN987654321",
            RegistrationDate = DateTime.UtcNow,
            CurrentMileage = 0m,
            FuelConsumption = 9.2m
        };

        _unitOfWorkMock.Setup(x => x.Vehicles.GetVehicleByPlateNumberAsync(createDto.PlateNumber, It.IsAny<CancellationToken>()))
            .ReturnsAsync((Vehicle?)null);

        _unitOfWorkMock.Setup(x => x.Vehicles.AddAsync(It.IsAny<Vehicle>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync((Vehicle v, CancellationToken ct) => v);

        _unitOfWorkMock.Setup(x => x.SaveChangesAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(1);

        // Act
        var result = await _vehicleService.CreateAsync(createDto);

        // Assert
        result.Should().NotBeNull();
        result.PlateNumber.Should().Be("XYZ-5678");
        result.Brand.Should().Be("Mercedes");
        result.Status.Should().Be(VehicleStatus.Available);
        _unitOfWorkMock.Verify(x => x.Vehicles.AddAsync(It.IsAny<Vehicle>(), It.IsAny<CancellationToken>()), Times.Once);
        _unitOfWorkMock.Verify(x => x.SaveChangesAsync(It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task CreateAsync_DuplicatePlateNumber_ThrowsInvalidOperationException()
    {
        // Arrange
        var createDto = new CreateVehicleDto
        {
            PlateNumber = "DUP-1234",
            Brand = "Scania",
            Model = "R450",
            Year = 2023,
            Type = VehicleType.Truck,
            LoadCapacity = 28000m,
            VinNumber = "VN111111111",
            RegistrationDate = DateTime.UtcNow,
            CurrentMileage = 1000m,
            FuelConsumption = 8.8m
        };

        var existingVehicle = new Vehicle
        {
            Id = Guid.NewGuid(),
            PlateNumber = "DUP-1234",
            Brand = "Volvo",
            Model = "FH",
            VinNumber = "VN222222222"
        };

        _unitOfWorkMock.Setup(x => x.Vehicles.GetVehicleByPlateNumberAsync(createDto.PlateNumber, It.IsAny<CancellationToken>()))
            .ReturnsAsync(existingVehicle);

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(
            async () => await _vehicleService.CreateAsync(createDto));

        _unitOfWorkMock.Verify(x => x.Vehicles.AddAsync(It.IsAny<Vehicle>(), It.IsAny<CancellationToken>()), Times.Never);
        _unitOfWorkMock.Verify(x => x.SaveChangesAsync(It.IsAny<CancellationToken>()), Times.Never);
    }

    [Fact]
    public async Task GetAvailableVehiclesAsync_ReturnsOnlyAvailableVehicles()
    {
        // Arrange
        var vehicles = new List<Vehicle>
        {
            new Vehicle { Id = Guid.NewGuid(), PlateNumber = "AAA-1111", Status = VehicleStatus.Available, Brand = "Volvo", Model = "FH", VinNumber = "VN1" },
            new Vehicle { Id = Guid.NewGuid(), PlateNumber = "BBB-2222", Status = VehicleStatus.Available, Brand = "Scania", Model = "R", VinNumber = "VN2" }
        };

        _unitOfWorkMock.Setup(x => x.Vehicles.GetAvailableVehiclesAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(vehicles);

        // Act
        var result = await _vehicleService.GetAvailableVehiclesAsync();

        // Assert
        result.Should().NotBeNull();
        result.Should().HaveCount(2);
        result.Should().OnlyContain(v => v.Status == VehicleStatus.Available);
    }

    [Fact]
    public async Task DeleteAsync_ExistingVehicle_DeletesSuccessfully()
    {
        // Arrange
        var vehicleId = Guid.NewGuid();
        var vehicle = new Vehicle
        {
            Id = vehicleId,
            PlateNumber = "DEL-9999",
            Brand = "Mercedes",
            Model = "Actros",
            VinNumber = "VN999999999"
        };

        _unitOfWorkMock.Setup(x => x.Vehicles.GetByIdAsync(vehicleId, It.IsAny<CancellationToken>()))
            .ReturnsAsync(vehicle);

        _unitOfWorkMock.Setup(x => x.Vehicles.DeleteAsync(vehicleId, It.IsAny<CancellationToken>()))
            .Returns(Task.CompletedTask);

        _unitOfWorkMock.Setup(x => x.SaveChangesAsync(It.IsAny<CancellationToken>()))
            .ReturnsAsync(1);

        // Act
        await _vehicleService.DeleteAsync(vehicleId);

        // Assert
        _unitOfWorkMock.Verify(x => x.Vehicles.DeleteAsync(vehicleId, It.IsAny<CancellationToken>()), Times.Once);
        _unitOfWorkMock.Verify(x => x.SaveChangesAsync(It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task DeleteAsync_NonExistingVehicle_ThrowsKeyNotFoundException()
    {
        // Arrange
        var vehicleId = Guid.NewGuid();

        _unitOfWorkMock.Setup(x => x.Vehicles.GetByIdAsync(vehicleId, It.IsAny<CancellationToken>()))
            .ReturnsAsync((Vehicle?)null);

        // Act & Assert
        await Assert.ThrowsAsync<KeyNotFoundException>(
            async () => await _vehicleService.DeleteAsync(vehicleId));

        _unitOfWorkMock.Verify(x => x.Vehicles.DeleteAsync(It.IsAny<Guid>(), It.IsAny<CancellationToken>()), Times.Never);
    }
}
