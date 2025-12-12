using Abp.Zero.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;
using WarehouseManagement.Entities;

namespace WarehouseManagement.EntityFrameworkCore
{
    public class WarehouseManagementDbContext : AbpZeroDbContext<Tenant, Role, User, WarehouseManagementDbContext>
    {
        public DbSet<Warehouse> Warehouses { get; set; }
        public DbSet<WarehouseZone> WarehouseZones { get; set; }
        public DbSet<InventoryItem> InventoryItems { get; set; }
        public DbSet<InventoryLocation> InventoryLocations { get; set; }
        public DbSet<Order> Orders { get; set; }
        public DbSet<OrderLine> OrderLines { get; set; }
        public DbSet<PickingTask> PickingTasks { get; set; }

        public WarehouseManagementDbContext(DbContextOptions<WarehouseManagementDbContext> options)
            : base(options)
        {
        }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            modelBuilder.Entity<Warehouse>(w =>
            {
                w.HasIndex(x => new { x.TenantId, x.Code }).IsUnique();
                w.HasIndex(x => x.TenantId);
            });

            modelBuilder.Entity<WarehouseZone>(wz =>
            {
                wz.HasIndex(x => new { x.TenantId, x.Code }).IsUnique();
                wz.HasOne(x => x.Warehouse)
                  .WithMany(w => w.Zones)
                  .HasForeignKey(x => x.WarehouseId)
                  .OnDelete(DeleteBehavior.Restrict);
            });

            modelBuilder.Entity<InventoryItem>(ii =>
            {
                ii.HasIndex(x => new { x.TenantId, x.SKU }).IsUnique();
                ii.HasIndex(x => x.WarehouseId);
                ii.HasOne(x => x.Warehouse)
                  .WithMany(w => w.InventoryItems)
                  .HasForeignKey(x => x.WarehouseId)
                  .OnDelete(DeleteBehavior.Restrict);
            });

            modelBuilder.Entity<Order>(o =>
            {
                o.HasIndex(x => new { x.TenantId, x.OrderNumber }).IsUnique();
                o.HasIndex(x => new { x.TenantId, x.Status });
            });

            modelBuilder.Entity<OrderLine>(ol =>
            {
                ol.HasOne(x => x.Order)
                  .WithMany(o => o.OrderLines)
                  .HasForeignKey(x => x.OrderId)
                  .OnDelete(DeleteBehavior.Cascade);
            });

            modelBuilder.Entity<PickingTask>(pt =>
            {
                pt.HasIndex(x => new { x.TenantId, x.Status });
                pt.HasIndex(x => x.AssignedUserId);
            });
        }
    }

    public class Tenant : Abp.MultiTenancy.AbpTenant<User>
    {
        protected Tenant()
        {
        }

        public Tenant(string tenancyName, string name)
            : base(tenancyName, name)
        {
        }
    }

    public class Role : Abp.Authorization.Roles.AbpRole<User>
    {
        protected Role()
        {
        }

        public Role(int? tenantId, string displayName)
            : base(tenantId, displayName)
        {
        }

        public Role(int? tenantId, string name, string displayName)
            : base(tenantId, name, displayName)
        {
        }
    }

    public class User : Abp.Authorization.Users.AbpUser<User>
    {
        public const string DefaultPassword = "123qwe";

        public static string CreateRandomPassword()
        {
            return Guid.NewGuid().ToString("N").Truncate(16);
        }

        public static User CreateTenantAdminUser(int tenantId, string emailAddress)
        {
            var user = new User
            {
                TenantId = tenantId,
                UserName = AdminUserName,
                Name = AdminUserName,
                Surname = AdminUserName,
                EmailAddress = emailAddress,
                Roles = new List<Abp.Authorization.Users.UserRole>()
            };

            user.SetNormalizedNames();

            return user;
        }
    }
}
