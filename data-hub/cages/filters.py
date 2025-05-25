from django_filters import rest_framework as filters
from .models import *


# class NodeFilter(filters.FilterSet):
#     class Meta:
#         model = Node
#         fields = ['name','type','status']


# class StoreRunFilter(filters.FilterSet):
#     class Meta:
#         model = Store_run
#         fields = ['name','type']


# class StatusRunFilter(filters.FilterSet):
#     #min_accuracy = filters.NumberFilter(field_name="store_run__initial_accuracy", lookup_expr='gte')
#     class Meta:
#         model = Status_run
#         fields = ['status','node__name']

# class HistoricalDetectionsFilter(filters.FilterSet):
#     #min_accuracy = filters.NumberFilter(field_name="store_run__initial_accuracy", lookup_expr='gte')
#     start_date = filters.DateTimeFilter(field_name="date_time", lookup_expr='gte')
#     end_date = filters.DateTimeFilter(field_name="date_time", lookup_expr='lte')
#     class Meta:
#         model = Historical_detections
#         fields = ['status_run__node__name','status_run__store_run__run_id','id']

# class HistoricalLogsFilter(filters.FilterSet):
#     start_date = filters.DateTimeFilter(field_name="date_time", lookup_expr='gte')
#     end_date  = filters.DateTimeFilter(field_name="date_time", lookup_expr='lte')
#     class Meta:
#         model = Historical_logs
#         fields = ['node_name','msg_type']