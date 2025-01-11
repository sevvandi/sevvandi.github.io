library(rmarkdown)

# Use absolute paths
project_dir <- getwd() #"."  # Replace with your actual path
rmd_dir <- file.path(project_dir, "_Rmd")
posts_dir <- file.path(project_dir, "_posts")

# Check if directories exist
if (!dir.exists(rmd_dir)) {
  stop("_Rmd directory not found at: ", rmd_dir)
}
if (!dir.exists(posts_dir)) {
  stop("_posts directory not found at: ", posts_dir)
}

# Get list of Rmd files
rmd_files <- list.files(rmd_dir, pattern = "\\.Rmd$", full.names = TRUE)

if (length(rmd_files) == 0) {
  stop("No .Rmd files found in: ", rmd_dir)
}

# Process each file
for (file in rmd_files) {
  output_file <- file.path(posts_dir, 
                           gsub("\\.Rmd$", ".md", basename(file)))
  render(file, output_file = output_file)
}
