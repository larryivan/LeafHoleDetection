# --- 加载所有需要的包 ---
library(sf)
library(ggplot2)
library(patchwork)
library(mapchina)
library(ggspatial)

# --- 准备数据 ---

# 1. 从 mapchina 包中明确地获取地图数据
# 使用 mapchina:: 的方式可以确保我们获取到正确的数据对象
china_map <- mapchina::china 
nine_line <- mapchina::nine  # <--- 这是关键的修正

# 2. 创建天目山坐标数据
# 经搜索，天目山位于浙江省杭州市临安区，大致坐标为东经119.4度，北纬30.3度. [2, 3]
tianmu_mountain <- data.frame(
  name = "天目山",
  lon = 119.43,
  lat = 30.34
)

# --- 绘制主图 ---
plot_main <- ggplot() +
  # 绘制中国地图
  geom_sf(data = china_map, fill = "#FFF5E1", color = "grey50", size = 0.5) +
  
  # 标注天目山（红点）
  geom_point(data = tianmu_mountain, aes(x = lon, y = lat), color = "red", size = 4, shape = 19) +
  
  # 在红点旁添加文字标签
  geom_text(data = tianmu_mountain, aes(x = lon + 2, y = lat + 2), label = "天目山", 
            color = "red", fontface = "bold", size = 4) +
  
  # 设置地图投影和显示范围
  coord_sf(crs = "+proj=longlat +datum=WGS84", 
           xlim = c(73, 136), ylim = c(18, 54), expand = FALSE) +
  
  # 添加指北针和比例尺
  annotation_scale(location = "bl", width_hint = 0.3, style="ticks", pad_x = unit(1, "cm")) +
  annotation_north_arrow(location = "tr", which_north = "true", 
                         pad_x = unit(0.2, "in"), pad_y = unit(0.2, "in"),
                         style = north_arrow_fancy_orienteering) +
  
  # 设置主题
  labs(title = "中国地图", x = "经度", y = "纬度") +
  theme_bw() +
  theme(
    panel.background = element_rect(fill = "lightblue"),
    panel.grid.major = element_line(color = "grey80", linetype = "dashed"),
    axis.text = element_text(size = 8)
  )

# --- 绘制南海小图 ---
plot_inset <- ggplot() +
  # 绘制地图数据（为了获取南海诸岛）
  geom_sf(data = china_map, fill = "#FFF5E1", color = "grey50") +
  
  # 绘制九段线
  geom_sf(data = nine_line, color = "red", size = 1) +
  
  # 设置小图的显示范围
  coord_sf(crs = "+proj=longlat +datum=WGS84",
           xlim = c(105, 122), ylim = c(0, 25), expand = FALSE) +
  
  # 使用极简主题并添加边框
  theme_void() +
  theme(
    panel.border = element_rect(colour = "black", fill = NA, size = 1),
    panel.background = element_rect(fill = "lightblue")
  )

# --- 组合并显示地图 ---
final_plot <- plot_main + inset_element(
  plot_inset, 
  left = 0.82,   # 调整小图左右位置
  bottom = 0.05, # 调整小图上下位置
  right = 1, 
  top = 0.3
)

print(final_plot)