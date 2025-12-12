using Abp.Application.Services;
using Abp.Authorization;
using Abp.Domain.Repositories;
using Abp.Linq.Extensions;
using Microsoft.EntityFrameworkCore;
using WarehouseManagement.Authorization;
using WarehouseManagement.Entities;
using WarehouseManagement.Inventory.Dto;

namespace WarehouseManagement.Inventory
{
    [AbpAuthorize(PermissionNames.Pages_Inventory)]
    public class InventoryAppService : AsyncCrudAppService<InventoryItem, InventoryItemDto, long, PagedInventoryResultRequestDto, CreateInventoryItemDto, InventoryItemDto>, IInventoryAppService
    {
        private readonly IRepository<InventoryItem, long> _inventoryRepository;
        private readonly IRepository<Warehouse, long> _warehouseRepository;

        public InventoryAppService(
            IRepository<InventoryItem, long> repository,
            IRepository<Warehouse, long> warehouseRepository)
            : base(repository)
        {
            _inventoryRepository = repository;
            _warehouseRepository = warehouseRepository;

            CreatePermissionName = PermissionNames.Pages_Inventory_Create;
            UpdatePermissionName = PermissionNames.Pages_Inventory_Edit;
            DeletePermissionName = PermissionNames.Pages_Inventory_Delete;
        }

        protected override IQueryable<InventoryItem> CreateFilteredQuery(PagedInventoryResultRequestDto input)
        {
            var query = Repository.GetAllIncluding(x => x.Warehouse, x => x.Location);

            if (input.WarehouseId.HasValue)
            {
                query = query.Where(x => x.WarehouseId == input.WarehouseId.Value);
            }

            if (!string.IsNullOrWhiteSpace(input.Keyword))
            {
                query = query.Where(x =>
                    x.SKU.Contains(input.Keyword) ||
                    x.ProductName.Contains(input.Keyword));
            }

            if (input.NeedsReorder.HasValue && input.NeedsReorder.Value)
            {
                query = query.Where(x => (x.QuantityOnHand - x.QuantityReserved) <= x.ReorderPoint);
            }

            return query;
        }

        public async Task<List<InventoryItemDto>> GetItemsNeedingReorderAsync()
        {
            var items = await _inventoryRepository
                .GetAllIncluding(x => x.Warehouse)
                .Where(x => (x.QuantityOnHand - x.QuantityReserved) <= x.ReorderPoint)
                .ToListAsync();

            return ObjectMapper.Map<List<InventoryItemDto>>(items);
        }

        public async Task<List<InventoryItemDto>> GetItemsByWarehouseAsync(long warehouseId)
        {
            var items = await _inventoryRepository
                .GetAllIncluding(x => x.Location)
                .Where(x => x.WarehouseId == warehouseId)
                .ToListAsync();

            return ObjectMapper.Map<List<InventoryItemDto>>(items);
        }

        [AbpAuthorize(PermissionNames.Pages_Inventory_AdjustQuantity)]
        public async Task AdjustQuantityAsync(AdjustInventoryQuantityInput input)
        {
            var item = await _inventoryRepository.GetAsync(input.InventoryItemId);

            item.QuantityOnHand += input.QuantityAdjustment;

            if (item.QuantityOnHand < 0)
            {
                throw new Abp.UI.UserFriendlyException("Quantity cannot be negative");
            }

            await _inventoryRepository.UpdateAsync(item);
        }

        public async Task<InventoryItemDto> GetItemBySKUAsync(string sku)
        {
            var item = await _inventoryRepository
                .GetAllIncluding(x => x.Warehouse, x => x.Location)
                .FirstOrDefaultAsync(x => x.SKU == sku);

            if (item == null)
            {
                throw new Abp.UI.UserFriendlyException($"Item with SKU {sku} not found");
            }

            return ObjectMapper.Map<InventoryItemDto>(item);
        }
    }
}
