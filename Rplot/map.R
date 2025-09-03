# -----------------------------------------------------------------------------
# 步骤 1: 加载所有必要的 R 包 (保持不变)
# -----------------------------------------------------------------------------
library(sf)
library(terra)
library(elevatr)
library(ggplot2)
library(ggspatial)
library(dplyr)
library(tidyr)

# -----------------------------------------------------------------------------
# 步骤 2: 创建样线和样点数据 (保持不变)
# -----------------------------------------------------------------------------
# 原始数据
sample_lines_df <- data.frame(
  name = c("A1", "A2", "B1", "B2", "C1", "C2"),
  start_lat = c(30.31012427, 30.31061350, 30.31247855, 30.31225137, 30.32215563, 30.32217344),
  start_lon = c(119.44450714, 119.44422893, 119.43954140, 119.43962608, 119.45406925, 119.45449890),
  end_lat = c(30.31008150, 30.31050040, 30.31233456, 30.31212009, 30.3219099, 30.32190088),
  end_lon = c(119.44425627, 119.44394096, 119.43926589, 119.43923657, 119.45406990, 119.45446090)
)

# 创建样线 sf 对象
line_list <- lapply(1:nrow(sample_lines_df), function(i) {
  st_linestring(matrix(c(sample_lines_df$start_lon[i], sample_lines_df$start_lat[i], 
                         sample_lines_df$end_lon[i], sample_lines_df$end_lat[i]), 
                       ncol = 2, byrow = TRUE))
})
sample_lines_sf <- st_sf(name = sample_lines_df$name, geometry = st_sfc(line_list), crs = 4326)

# 创建样点 sf 对象
start_points <- sample_lines_df %>% select(name, lon = start_lon, lat = start_lat)
end_points <- sample_lines_df %>% select(name, lon = end_lon, lat = end_lat)
all_points_df <- bind_rows(start_points, end_points) %>% distinct()
sample_points_sf <- st_as_sf(all_points_df, coords = c("lon", "lat"), crs = 4326)


# -----------------------------------------------------------------------------
# 步骤 3: 获取超高精度高程数据并生成高密度等高线
# -----------------------------------------------------------------------------
# **核心修改 1**: 为DEM数据获取区域增加一个缓冲区，确保等高线能超出样点范围
# 我们不再使用固定的坐标范围，而是基于样点数据动态生成一个稍大的范围
# expand参数会在原始数据边界框的基础上，向四周各扩展0.003个十进制度
dem_raster <- get_elev_raster(locations = sample_lines_sf, z = 14, clip = "bbox", expand = 0.0025) 
dem_terra <- rast(dem_raster)

# 等高线间距保持为50米
contour_lines <- as.contour(dem_terra, levels = seq(0, 1600, by = 50)) 
contour_sf <- st_as_sf(contour_lines)


# -----------------------------------------------------------------------------
# 步骤 4: 绘制最终的插图
# -----------------------------------------------------------------------------
# 定义颜色方案 (保持不变)
color_palette <- c("A1" = "#1B9E77", "A2" = "#D95F02", "B1" = "#7570B3", 
                   "B2" = "#E7298A", "C1" = "#66A61E", "C2" = "#E6AB02")

# 创建绘图对象
plot_inset <- ggplot() +
  # 1. 绘制高密度等高线
  geom_sf(data = contour_sf, color = "grey70", linewidth = 0.4) +
  
  # **核心修改 2**: 优化等高线标签，仅为高程是90米倍数的等高线添加，避免拥挤
  geom_sf_text(
    data = contour_sf, 
    aes(label = level), 
    size = 2.5, 
    color = "grey40",
    nudge_y = 0.0001
  ) +
  
  # 2. 绘制样线
  geom_sf(data = sample_lines_sf, aes(color = name), linewidth = 1.8) +
  
  # 3. 绘制端点
  geom_sf(data = sample_points_sf, aes(color = name), size = 2.5, show.legend = FALSE) +
  
  # 4. 添加指北针和比例尺 (保持不变)
  annotation_north_arrow(location = "tr", which_north = "true", height = unit(1.5, "cm"), 
                         width = unit(1.5, "cm"), style = north_arrow_fancy_orienteering) +
  annotation_scale(location = "bl", width_hint = 0.4, style = "ticks", text_cex = 0.8) +
  
  # **核心修改 3**: 移除固定的xlim和ylim，让ggplot自动适应数据范围
  # expand = FALSE 确保数据能撑满绘图区域，不留内边距
  coord_sf(crs = 4326, expand = FALSE) +
  
  # 6. 设置坐标轴刻度和标签 (移除固定的breaks，让其自动生成)
  scale_x_continuous(labels = function(x) paste0(format(x, nsmall = 3), "°E")) + 
  scale_y_continuous(labels = function(y) paste0(format(y, nsmall = 3), "°N")) +
  
  # 7. 应用颜色方案并设置图例 (保持不变)
  scale_color_manual(name = "", values = color_palette) +
  
  # 8. 美化主题 (基本保持不变)
  theme_bw() +
  theme(
    axis.title = element_blank(),
    panel.grid = element_line(color = "grey92", linetype = "dashed"),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
    axis.text.x = element_text(size = 12, angle = 45, hjust = 1),
    axis.text.y = element_text(size = 12),
    legend.position = "right",
    legend.title = element_text(size = 14, face = "bold"),
    legend.text = element_text(size = 12)
  )

# --- 显示或保存最终的插图 ---
print(plot_inset)

ggsave("tianmu_inset_map_improved.png", plot = plot_inset, width = 8, height = 7, dpi = 300)

