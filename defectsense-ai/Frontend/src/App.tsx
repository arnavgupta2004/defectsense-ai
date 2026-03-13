import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes, Navigate } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import Navbar from "@/components/Navbar";
import Dashboard from "@/pages/Dashboard";
import Inspect from "@/pages/Inspect";
import Results from "@/pages/Results";
import HistoryPage from "@/pages/HistoryPage";
import ModelPage from "@/pages/ModelPage";
import NotFound from "@/pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Sonner />
      <BrowserRouter>
        <div className="min-h-screen bg-background grid-bg">
          <Navbar />
          <main className="pt-14">
            <div className="max-w-screen-2xl mx-auto p-6">
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/inspect" element={<Inspect />} />
                <Route path="/results/:id" element={<Results />} />
                <Route path="/history" element={<HistoryPage />} />
                <Route path="/model" element={<ModelPage />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </div>
          </main>
        </div>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
