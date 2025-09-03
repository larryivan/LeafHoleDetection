const translations = {
    en: {
        title: "Leaf Hole Detection System",
        subtitle: "Leaf analysis with manual editing",
        singleMode: "Single Image",
        batchMode: "Batch Processing",
        uploadTitle: "Drag & Drop your leaf image here",
        uploadSubtitle: "or click to select a file",
        batchUploadTitle: "Drag & Drop multiple leaf images here",
        batchUploadSubtitle: "or click to select multiple files",
        chooseFile: "Choose File",
        chooseFiles: "Choose Files",
        uploadInfo: "Supported formats: PNG, JPG, JPEG, GIF, BMP (Max 16MB)",
        referenceInfo: "Include a 1cm×1cm reference square for absolute area measurements",
        processing: "Processing Image...",
        processingSubtitle: "AI is analyzing your leaf image",
        resultsTitle: "Detection Results",
        holeRatio: "Hole Ratio",
        leafArea: "Leaf Area",
        holeArea: "Hole Area",
        holeCount: "Hole Count",
        leafAreaCm2: "Leaf Area (cm²)",
        holeAreaCm2: "Hole Area (cm²)",
        leafLength: "Leaf Length",
        leafWidth: "Leaf Width",
        leafLengthCm: "Leaf Length (cm)",
        leafWidthCm: "Leaf Width (cm)",
        referenceDetected: "Reference square detected",
        pixelsPerCm: "pixels per cm",
        manualEdit: "Manual Edit",
        newAnalysis: "New Analysis",
        originalImage: "Original Image",
        enhancedImage: "Enhanced Image",
        leafSegmentation: "Leaf Segmentation",
        detectedHoles: "Detected Holes",
        finalResult: "Final Result",
        batchResultsTitle: "Batch Processing Results",
        totalFiles: "Total Files",
        successfulFiles: "Successful",
        avgHoleRatio: "Avg Hole Ratio",
        filesWithReference: "With Reference",
        totalLeafAreaCm2: "Total Leaf Area (cm²)",
        totalHoleAreaCm2: "Total Hole Area (cm²)",
        overallHoleRatioCm2: "Overall Hole Ratio",
        exportResults: "Export Results (CSV)",
        filename: "Filename",
        success: "Success",
        manualEditTitle: "Manual Editing Mode",
        manualEditDesc: "Click and drag to select leaf area or holes. Use the tools below to switch between modes.",
        selectLeafArea: "Select Leaf Area",
        markHoles: "Mark Holes",
        clearAll: "Clear All",
        recalculate: "Recalculate",
        cancel: "Cancel",
        canvasInfo: "Left click and drag to draw selections. Switch tools to mark different areas.",
        developer: "Developed by",
        contact: "Contact:",
        fileSizeError: "File size must be less than 16MB",
        invalidFileError: "Please select an image file",
        processingError: "Processing failed",
        recalculatedSuccess: "Results updated successfully!",
        processImagesPending: "Processing images will be displayed here",
        processImagesNote: "Images will be shown after batch processing is complete",
        batchEditPlaceholder: "Batch item editing will be available soon"
    },
    zh: {
        title: "叶片虫洞检测系统",
        subtitle: "叶片分析与手动编辑",
        singleMode: "单张图像",
        batchMode: "批量处理",
        uploadTitle: "拖拽叶片图像到此处",
        uploadSubtitle: "或点击选择文件",
        batchUploadTitle: "拖拽多张叶片图像到此处",
        batchUploadSubtitle: "或点击选择多个文件",
        chooseFile: "选择文件",
        chooseFiles: "选择文件",
        uploadInfo: "支持格式：PNG, JPG, JPEG, GIF, BMP (最大16MB)",
        referenceInfo: "包含1cm×1cm参照方块可获得绝对面积测量",
        processing: "正在处理图像...",
        processingSubtitle: "AI正在分析您的叶片图像",
        resultsTitle: "检测结果",
        holeRatio: "虫洞占比",
        leafArea: "叶片面积",
        holeArea: "虫洞面积",
        holeCount: "虫洞数量",
        leafAreaCm2: "叶片面积 (cm²)",
        holeAreaCm2: "虫洞面积 (cm²)",
        leafLength: "叶片长度",
        leafWidth: "叶片宽度",
        leafLengthCm: "叶片长度 (cm)",
        leafWidthCm: "叶片宽度 (cm)",
        referenceDetected: "检测到参照方块",
        pixelsPerCm: "像素每厘米",
        manualEdit: "手动编辑",
        newAnalysis: "重新分析",
        originalImage: "原始图像",
        enhancedImage: "增强图像",
        leafSegmentation: "叶片分割",
        detectedHoles: "检测到的虫洞",
        finalResult: "最终结果",
        batchResultsTitle: "批量处理结果",
        totalFiles: "总文件数",
        successfulFiles: "成功处理",
        avgHoleRatio: "平均虫洞占比",
        filesWithReference: "含参照方块",
        totalLeafAreaCm2: "总叶片面积 (cm²)",
        totalHoleAreaCm2: "总虫洞面积 (cm²)",
        overallHoleRatioCm2: "总体虫洞占比",
        exportResults: "导出结果 (CSV)",
        filename: "文件名",
        success: "成功",
        manualEditTitle: "手动编辑模式",
        manualEditDesc: "点击并拖拽选择叶片区域或虫洞。使用下方工具切换模式。",
        selectLeafArea: "选择叶片区域",
        markHoles: "标记虫洞",
        clearAll: "清除全部",
        recalculate: "重新计算",
        cancel: "取消",
        canvasInfo: "左键点击并拖拽绘制选择区域。切换工具标记不同区域。",
        developer: "开发者",
        contact: "联系方式：",
        fileSizeError: "文件大小必须小于16MB",
        invalidFileError: "请选择图像文件",
        processingError: "处理失败",
        recalculatedSuccess: "结果更新成功！",
        processImagesPending: "处理图像将在此处显示",
        processImagesNote: "批量处理完成后将显示图像",
        batchEditPlaceholder: "批量项目编辑功能即将推出"
    }
};

let currentLang = localStorage.getItem('language') || 'en';

function t(key) {
    const keys = key.split('.');
    let value = translations[currentLang];
    
    for (const k of keys) {
        value = value[k];
        if (!value) break;
    }
    
    return value || translations['en'][key] || key;
}

function setLanguage(lang) {
    currentLang = lang;
    localStorage.setItem('language', lang);
    updateUI();
    updateLanguageButtons();
}

function updateLanguageButtons() {
    document.querySelectorAll('.lang-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.lang === currentLang);
    });
}

function updateUI() {
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        const translation = t(key);
        element.textContent = translation;
    });
}

document.addEventListener('DOMContentLoaded', function() {
    updateUI();
    updateLanguageButtons();
});

function updateUI() {
    // Update all elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(element => {
        const key = element.getAttribute('data-i18n');
        const translation = t(key);
        
        if (element.tagName === 'INPUT' && element.type === 'button') {
            element.value = translation;
        } else if (element.hasAttribute('placeholder')) {
            element.placeholder = translation;
        } else {
            element.textContent = translation;
        }
    });
}

// Initialize language on page load
document.addEventListener('DOMContentLoaded', function() {
    updateUI();
    updateLanguageButtons();
});
