using Abp.Authorization;
using Abp.Localization;

namespace WarehouseManagement.Authorization
{
    public class WarehouseManagementAuthorizationProvider : AuthorizationProvider
    {
        public override void SetPermissions(IPermissionDefinitionContext context)
        {
            context.CreatePermission(PermissionNames.Pages_Tenants, L("Tenants"));
            context.CreatePermission(PermissionNames.Pages_Users, L("Users"));
            context.CreatePermission(PermissionNames.Pages_Roles, L("Roles"));

            var warehouses = context.CreatePermission(PermissionNames.Pages_Warehouses, L("Warehouses"));
            warehouses.CreateChildPermission(PermissionNames.Pages_Warehouses_Create, L("CreateNewWarehouse"));
            warehouses.CreateChildPermission(PermissionNames.Pages_Warehouses_Edit, L("EditWarehouse"));
            warehouses.CreateChildPermission(PermissionNames.Pages_Warehouses_Delete, L("DeleteWarehouse"));

            var inventory = context.CreatePermission(PermissionNames.Pages_Inventory, L("Inventory"));
            inventory.CreateChildPermission(PermissionNames.Pages_Inventory_Create, L("CreateInventoryItem"));
            inventory.CreateChildPermission(PermissionNames.Pages_Inventory_Edit, L("EditInventoryItem"));
            inventory.CreateChildPermission(PermissionNames.Pages_Inventory_Delete, L("DeleteInventoryItem"));
            inventory.CreateChildPermission(PermissionNames.Pages_Inventory_AdjustQuantity, L("AdjustInventoryQuantity"));

            var orders = context.CreatePermission(PermissionNames.Pages_Orders, L("Orders"));
            orders.CreateChildPermission(PermissionNames.Pages_Orders_Create, L("CreateOrder"));
            orders.CreateChildPermission(PermissionNames.Pages_Orders_Edit, L("EditOrder"));
            orders.CreateChildPermission(PermissionNames.Pages_Orders_Cancel, L("CancelOrder"));
            orders.CreateChildPermission(PermissionNames.Pages_Orders_Process, L("ProcessOrder"));

            var picking = context.CreatePermission(PermissionNames.Pages_Picking, L("Picking"));
            picking.CreateChildPermission(PermissionNames.Pages_Picking_Assign, L("AssignPickingTask"));
            picking.CreateChildPermission(PermissionNames.Pages_Picking_Complete, L("CompletePickingTask"));

            var reports = context.CreatePermission(PermissionNames.Pages_Reports, L("Reports"));
            reports.CreateChildPermission(PermissionNames.Pages_Reports_Inventory, L("InventoryReports"));
            reports.CreateChildPermission(PermissionNames.Pages_Reports_Orders, L("OrderReports"));
            reports.CreateChildPermission(PermissionNames.Pages_Reports_Performance, L("PerformanceReports"));
        }

        private static ILocalizableString L(string name)
        {
            return new LocalizableString(name, WarehouseManagementConsts.LocalizationSourceName);
        }
    }
}
