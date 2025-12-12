FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ["src/FleetManagement.Web/FleetManagement.Web.csproj", "src/FleetManagement.Web/"]
COPY ["src/FleetManagement.Application/FleetManagement.Application.csproj", "src/FleetManagement.Application/"]
COPY ["src/FleetManagement.Domain/FleetManagement.Domain.csproj", "src/FleetManagement.Domain/"]
COPY ["src/FleetManagement.Infrastructure/FleetManagement.Infrastructure.csproj", "src/FleetManagement.Infrastructure/"]
COPY ["src/FleetManagement.Shared/FleetManagement.Shared.csproj", "src/FleetManagement.Shared/"]
RUN dotnet restore "src/FleetManagement.Web/FleetManagement.Web.csproj"
COPY . .
WORKDIR "/src/src/FleetManagement.Web"
RUN dotnet build "FleetManagement.Web.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "FleetManagement.Web.csproj" -c Release -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "FleetManagement.Web.dll"]
