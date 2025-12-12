using AutoMapper;
using FleetManagement.Application.DTOs;
using FleetManagement.Domain.Entities;

namespace FleetManagement.Application.Mappings;

public class MappingProfile : Profile
{
    public MappingProfile()
    {
        CreateMap<Vehicle, VehicleDto>()
            .ForMember(dest => dest.CurrentDriverName,
                opt => opt.MapFrom(src => src.CurrentDriver != null ? src.CurrentDriver.FullName : null));

        CreateMap<CreateVehicleDto, Vehicle>();
        CreateMap<UpdateVehicleDto, Vehicle>();

        CreateMap<Driver, DriverDto>();
        CreateMap<CreateDriverDto, Driver>()
            .ForMember(dest => dest.Status, opt => opt.MapFrom(src => Domain.Enums.DriverStatus.Active));

        CreateMap<Route, RouteDto>()
            .ForMember(dest => dest.VehiclePlateNumber,
                opt => opt.MapFrom(src => src.Vehicle.PlateNumber))
            .ForMember(dest => dest.DriverName,
                opt => opt.MapFrom(src => src.Driver.FullName));

        CreateMap<CreateRouteDto, Route>()
            .ForMember(dest => dest.RouteNumber, opt => opt.MapFrom(src => GenerateRouteNumber()))
            .ForMember(dest => dest.Status, opt => opt.MapFrom(src => Domain.Enums.RouteStatus.Scheduled));
    }

    private static string GenerateRouteNumber()
    {
        return $"RT-{DateTime.UtcNow:yyyyMMdd}-{Guid.NewGuid().ToString().Substring(0, 8).ToUpper()}";
    }
}
