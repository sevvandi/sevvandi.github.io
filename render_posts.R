library(rmarkdown)

# Get list of Rmd files in _Rmd directory
rmd_files <- list.files("_Rmd", pattern = "*.Rmd", full.names = TRUE)

# Render each Rmd file to md in _posts directory
for (file in rmd_files) {
  output_file <- file.path("_posts", 
                           gsub(".Rmd$", ".md", basename(file)))
  render(file, output_file = output_file)
}