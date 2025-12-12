using Abp.Application.Services.Dto;
using Shouldly;
using System.Threading.Tasks;
using WarehouseManagement.Entities;
using WarehouseManagement.Inventory;
using WarehouseManagement.Inventory.Dto;
using Xunit;

namespace WarehouseManagement.Tests.Inventory
{
    public class InventoryAppService_Tests : WarehouseManagementTestBase
    {
        private readonly IInventoryAppService _inventoryAppService;

        public InventoryAppService_Tests()
        {
            _inventoryAppService = Resolve<IInventoryAppService>();
        }

        [Fact]
        public async Task Should_Get_All_Inventory_Items()
        {
            var output = await _inventoryAppService.GetAllAsync(new PagedInventoryResultRequestDto());

            output.Items.Count.ShouldBeGreaterThan(0);
        }

        [Fact]
        public async Task Should_Create_Inventory_Item()
        {
            var warehouse = await CreateWarehouseAsync();

            var input = new CreateInventoryItemDto
            {
                SKU = "TEST-001",
                ProductName = "Test Product",
                Description = "Test Description",
                WarehouseId = warehouse.Id,
                QuantityOnHand = 100,
                ReorderPoint = 20,
                ReorderQuantity = 50,
                UnitCost = 10.50m
            };

            var result = await _inventoryAppService.CreateAsync(input);

            result.ShouldNotBeNull();
            result.SKU.ShouldBe("TEST-001");
            result.QuantityOnHand.ShouldBe(100);
        }

        [Fact]
        public async Task Should_Get_Items_Needing_Reorder()
        {
            var warehouse = await CreateWarehouseAsync();
            await CreateLowStockItemAsync(warehouse.Id);

            var items = await _inventoryAppService.GetItemsNeedingReorderAsync();

            items.Count.ShouldBeGreaterThan(0);
            items.ShouldContain(x => x.NeedsReorder);
        }

        [Fact]
        public async Task Should_Adjust_Inventory_Quantity()
        {
            var warehouse = await CreateWarehouseAsync();
            var item = await CreateInventoryItemAsync(warehouse.Id, initialQuantity: 50);

            await _inventoryAppService.AdjustQuantityAsync(new AdjustInventoryQuantityInput
            {
                InventoryItemId = item.Id,
                QuantityAdjustment = 10,
                Reason = "Test adjustment"
            });

            var updated = await _inventoryAppService.GetAsync(new EntityDto<long>(item.Id));
            updated.QuantityOnHand.ShouldBe(60);
        }

        [Fact]
        public async Task Should_Get_Item_By_SKU()
        {
            var warehouse = await CreateWarehouseAsync();
            await CreateInventoryItemAsync(warehouse.Id, sku: "UNIQUE-SKU-123");

            var item = await _inventoryAppService.GetItemBySKUAsync("UNIQUE-SKU-123");

            item.ShouldNotBeNull();
            item.SKU.ShouldBe("UNIQUE-SKU-123");
        }

        private async Task<Warehouse> CreateWarehouseAsync()
        {
            return await UsingDbContextAsync(async context =>
            {
                var warehouse = new Warehouse
                {
                    Code = $"WH-{Guid.NewGuid().ToString().Substring(0, 8)}",
                    Name = "Test Warehouse",
                    Address = "123 Test St",
                    City = "Test City",
                    State = "TS",
                    ZipCode = "12345",
                    Country = "Test Country",
                    TotalArea = 10000,
                    StorageCapacity = 5000,
                    IsActive = true,
                    TenantId = AbpSession.GetTenantId()
                };

                context.Warehouses.Add(warehouse);
                await context.SaveChangesAsync();
                return warehouse;
            });
        }

        private async Task<InventoryItem> CreateInventoryItemAsync(long warehouseId, string sku = null, int initialQuantity = 100)
        {
            return await UsingDbContextAsync(async context =>
            {
                var item = new InventoryItem
                {
                    SKU = sku ?? $"SKU-{Guid.NewGuid().ToString().Substring(0, 8)}",
                    ProductName = "Test Product",
                    WarehouseId = warehouseId,
                    QuantityOnHand = initialQuantity,
                    QuantityReserved = 0,
                    ReorderPoint = 20,
                    ReorderQuantity = 50,
                    UnitCost = 10.00m,
                    IsActive = true,
                    TenantId = AbpSession.GetTenantId()
                };

                context.InventoryItems.Add(item);
                await context.SaveChangesAsync();
                return item;
            });
        }

        private async Task<InventoryItem> CreateLowStockItemAsync(long warehouseId)
        {
            return await CreateInventoryItemAsync(warehouseId, initialQuantity: 10);
        }
    }
}
