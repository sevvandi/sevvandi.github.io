library(rmarkdown)

# Print working directory and check directory structure
print(paste("Working directory:", getwd()))
print(paste("_Rmd exists:", dir.exists("_Rmd")))

# Create _posts directory if it doesn't exist
if (!dir.exists("_posts")) {
  dir.create("_posts")
  print("Created _posts directory")
}

# Find all Rmd files
rmd_files <- list.files("_Rmd", pattern = "\\.Rmd$", full.names = TRUE)
print(paste("Rmd files found:", length(rmd_files)))
print(rmd_files)

# Only proceed if files were found
if (length(rmd_files) > 0) {
  for (file in rmd_files) {
    print(paste("Processing:", file))
    try({
      output_file <- file.path("_posts", 
                               gsub("\\.Rmd$", ".md", basename(file)))
      render(file, output_file = output_file)
    })
  }
} else {
  stop("No .Rmd files found in _Rmd directory")
}